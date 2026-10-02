"""Read-only audit of saved synthetic training data; no generation or training."""
import json
import random
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sml_v2.common import atomic_json, file_sha256, read_json
from sml_v2.tokenization import Tokenizer
from sft.teacher_chat_768_v1.data import turns, visible_prefix

RUN = ROOT / 'runs/teacher_chat_768_v1'
OUT = ROOT / 'diagnostics/teacher_data_audit_v1'
NUMBERS = dict(zip('one two three four five six seven eight nine ten'.split(), range(1, 11)))
EXAMPLES = {
    '8b2b4536acb6': 'Exactly two paragraphs: target has one; boolean pass contradicts review rationale.',
    '82186871a467': 'Second confirmed two-paragraph failure with the same review contradiction.',
    '253e8fba99df': 'Career advice becomes invented assistant autobiography; no user-requested roleplay.',
    '6a7d02427808': 'Generated follow-up asks the assistant about the user\'s business plans. Target acknowledges role confusion.',
    '8e124bd5e27d': 'Generated follow-up praises a nonexistent outline and refers to nonexistent Module 2.',
    '57e24f8e7b54': 'Follow-up invents a prior Elara description. Target correctly denies it but then merely offers to create a character.',
    'ea7800b0393f': 'Grounded category trains writing a passage and question, rather than answering from supplied evidence.',
    'ddfc8948fca2': 'Target misattributes JFK\'s famous 1961 inaugural line to a 1960 campaign slogan.',
    'b896679c29f6': 'Positive example: a repetitive draft is corrected into exactly three useful requested bullets.',
    '2de021f55afd': 'Positive example: directly completes a short text-message request.',
}


def repeated(ids):
    counts = Counter(tuple(ids[i:i+4]) for i in range(len(ids)-3))
    return bool(counts and sum(n-1 for n in counts.values()) / max(1, len(ids)-3) > .2)


def main():
    path = RUN / 'data/prepared.json'
    before = file_sha256(path)
    data = read_json(path)['data']
    teacher = [r for r in data['train'] if r['source'] == 'teacher']
    replay = [r for r in data['train'] if r['source'] == 'replay']
    follow = [r for r in teacher if r['kind'] == 'followup']
    tokenizer = Tokenizer(ROOT / 'tokenizer/bytebpe32k_v1')
    stats = dict(prepared_sha256=before, teacher_records=len(teacher),
                 teacher_groups=len(follow), replay_records=len(replay),
                 replay_mix=dict(Counter(r['original_source'] for r in replay)), splits={},
                 teacher_tokens=0, mask_and_eos_checks=0, repetitive_teacher_targets=0,
                 repetitive_draft_contexts=0, explicit_format_checks={}, explicit_format_failures=[])
    for split, rows in data.items():
        initial = [r for r in rows if r['source'] == 'teacher' and r['kind'] == 'initial']
        stats['splits'][split] = dict(teacher_groups=len(initial),
            sources=dict(Counter(r['origin']['source'] for r in initial)),
            with_system_message=sum(r['messages'][0]['role'] == 'system' for r in initial))
    for r in teacher:
        e = turns(tokenizer, r, 2048)
        head = tokenizer.encode(visible_prefix(r['messages'][:-1]))
        full = tokenizer.encode(visible_prefix(r['messages'][:-1]) + ' ' + r['messages'][-1]['content'])
        assert len(e) == 1
        assert e[0]['y'][:len(head)-1] == [-100] * (len(head)-1)
        assert e[0]['y'][len(head)-1:] == full[len(head):] + [tokenizer.eos]
        stats['mask_and_eos_checks'] += 1
        stats['teacher_tokens'] += e[0]['targets']
        answer = r['messages'][-1]['content']
        stats['repetitive_teacher_targets'] += repeated(tokenizer.encode(answer))
        prompt = r['messages'][-2]['content']
        pattern = r'exactly (one|two|three|four|five|six|seven|eight|nine|ten|\d+) (?:short |concise )?(paragraphs?|bullet points?)'
        for m in re.finditer(pattern, prompt, re.I):
            value = m[1].lower()
            expected = NUMBERS[value] if value in NUMBERS else int(value)
            kind = 'paragraphs' if m[2].lower().startswith('paragraph') else 'bullets'
            actual = (len([s for s in re.split(r'\n\s*\n', answer.strip()) if s.strip()])
                      if kind == 'paragraphs' else len(re.findall(r'^\s*[-*•]\s+', answer, re.M)))
            stats['explicit_format_checks'][kind] = stats['explicit_format_checks'].get(kind, 0) + 1
            if actual != expected:
                stats['explicit_format_failures'].append(dict(group=r['group'], kind=kind,
                    expected=expected, actual=actual, row_id=r['id']))
    stats['repetitive_draft_contexts'] = sum(repeated(tokenizer.encode(r['messages'][1]['content'])) for r in follow)
    stats['synthetic_rows_with_system_message'] = sum(r['messages'][0]['role']=='system' for r in teacher)
    stats['replay_rows_with_system_message'] = sum(r['messages'][0]['role']=='system' for r in replay)
    grounded = [r for r in follow if r['category']=='grounded']
    stats['grounded_followups'] = len(grounded)
    stats['grounded_followups_starting_write_create_compose'] = sum(bool(re.match(
        r'(?:Please )?(?:write|create|compose)\b', r['messages'][-2]['content'], re.I)) for r in grounded)
    stats['target_length_tokens'] = {kind: dict(median=statistics.median(r['assistant_targets'] for r in teacher if r['kind']==kind),
        maximum=max(r['assistant_targets'] for r in teacher if r['kind']==kind)) for kind in ('initial','followup')}
    sample = random.Random(9292026).sample(follow, 32)
    stats['random_sample_group_ids'] = [r['group'] for r in sample]
    stats['scope_note'] = 'Structural checks cover all 2048 teacher targets. Semantic review sampled 32 groups plus targeted examples; this is not a complete factual or semantic certification.'
    examples = []
    for prefix, finding in EXAMPLES.items():
        r = next(r for r in follow if r['group'].startswith(prefix))
        raw = read_json(RUN / 'data/groups/train' / (r['group']+'.json'))
        examples.append(dict(finding=finding, row=r, saved_review=raw['result']['review'],
                             source_file=str(RUN / 'data/groups/train' / (r['group']+'.json'))))
    receipt = read_json(RUN / 'checkpoint_cleanup_20260929.json')
    for relative, digest in receipt['preserved_data_hashes'].items():
        assert file_sha256(RUN / relative) == digest
    assert file_sha256(path) == before
    stats['original_data_files_verified_unchanged'] = len(receipt['preserved_data_hashes'])
    atomic_json(OUT/'statistics.json', stats)
    atomic_json(OUT/'examples.json', examples)
    lines = ['# Synthetic data audit: concrete examples', '', stats['scope_note'], '']
    for i, ex in enumerate(examples, 1):
        r = ex['row']
        lines += [f'## {i}. {ex["finding"]}', '', f'Group: `{r["group"]}`; category: `{r["category"]}`', '']
        for m in r['messages']:
            lines += ['**'+m['role']+'**', '', m['content'], '']
        lines += ['**Saved final review**', '', '```json', json.dumps(ex['saved_review']['final_review'], indent=2), '```', '']
    (OUT/'examples.md').write_text('\n'.join(lines))
    print(json.dumps(stats, indent=2))


if __name__ == '__main__':
    main()
