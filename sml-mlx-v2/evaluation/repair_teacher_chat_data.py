#!/usr/bin/env python3
"""Apply documented human-authored repairs to a copy of teacher-chat data."""
import copy
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sml_v2.common import atomic_json, file_sha256, fingerprint, read_json

RUN = ROOT / 'runs/teacher_chat_768_v1'
SOURCE = RUN / 'data/prepared.json'
OUT = RUN / 'data_repaired_v1'
AUDIT = ROOT / 'diagnostics/teacher_data_audit_v1'
NUMBERS = dict(zip('one two three four five six seven eight nine ten'.split(), range(1, 11)))
FORMAT = re.compile(r'exactly (one|two|three|four|five|six|seven|eight|nine|ten|\d+) (?:short |concise )?(paragraphs?|bullet points?)', re.I)

REPAIRS = {
    '8b2b4536acb6fa99124b0dd30e341edf043ba38e0850d94e1e481f5b65dc88e5': {
        'followup_answer': "Community service has shaped the person I am becoming. Through [name a real volunteer role or project], I helped [describe a specific action and its effect on the community]. Taking responsibility for [a real leadership task or achievement, if applicable] taught me [lesson you learned] and strengthened my interest in [academic subject or career goal]. These experiences showed me how steady, practical service can make a difference.",
        'followup_answer_2': "This scholarship would help me continue serving my community while pursuing [degree or program] at [school]. It would reduce [specific financial barrier] and give me more time to [specific service activity or goal]. I hope to use what I learn in [field] to [future contribution]. My next steps are [brief, realistic plan], and I would be grateful for support as I work toward them.",
        'reason': 'Fixed the exactly-two-paragraph violation and replaced invented personal history with clear fill-in prompts.'},
    '82186871a4670bac5db3a63da0dfe73d5e92fa1db0c6495ee22e618100bc371e': {
        'followup_answer': "The first time I understood an animal, I was standing beside a creek when a fox looked straight at me and said, 'The water is rising.' I ran to warn the campers downstream, but no one believed me until the current swept away the footbridge. After that, I began hearing more than words: a crow's sharp alarm, a dog's lonely thoughts, a turtle's patient memories of the pond. The constant voices frightened me, and I worried I was losing touch with people. I kept practicing, learning to listen without mistaking every sound for a command. When a storm later trapped a fawn behind a fallen tree, I followed the animals' warnings and found a safe path through the woods. Helping it reach its mother showed me that the gift could serve others, as long as I respected the animals instead of trying to control them. I still have much to learn, but I no longer feel alone in the forest.",
        'followup_answer_2': "Over time, the animals taught me that understanding them did not mean always agreeing with them. I learned to ask before approaching, to give frightened creatures space, and to accept that some would choose not to speak. My greatest test came when hunters entered the forest. The deer warned me first, while the owls showed me a route to the ranger station. I reported what I had seen and guided the rangers to the animals without revealing every hidden den. The forest was protected, and the animals remained safe. I returned home tired but proud: I had become an animal whisperer by listening carefully and using my gift with care, not by making the animals obey.",
        'reason': 'Replaced a one-paragraph answer with a two-paragraph first-person story that follows the requested arc.'},
    '253e8fba99df': {
        'followup_prompt': "You said your corporate job leaves you feeling like you are going through the motions. What questions could you ask yourself this week to identify which tasks drain you and which parts still feel meaningful?",
        'followup_answer': "I should correct my previous reply: I do not have a personal workweek, and your earlier message was about your experience. To learn what is draining you, jot down a few moments this week and ask: What was I doing? Which part felt pointless or stressful? Did I have enough choice or support? Was there any part I found interesting? After several examples, look for a pattern. It may point to a particular task, team environment, or lack of autonomy, which gives you a smaller first change to try before making a major career decision.",
        'reason': 'Removed fabricated assistant autobiography and made the follow-up answer useful while respecting who supplied the career facts.'},
    '6a7d02427808': {
        'followup_prompt': "For my planned store selling reusable bags, refillable bottles, and eco-friendly cleaners, which e-commerce platforms should I compare, and what should I check before choosing a bottle supplier?",
        'followup_answer': "For a small catalog, compare Shopify, which bundles hosting and store tools, with WooCommerce, which gives you more control but requires you to manage hosting and extensions. For bottle suppliers, request samples and documentation for the bottle material, food-contact safety, leak testing, recycled-content claims, minimum order, lead time, and return terms. Check test reports and certifications directly, and compare total landed cost before committing.",
        'reason': 'Rephrased a question addressed to the assistant as a useful follow-up about the user’s stated store, then answered it directly.'},
    '8e124bd5e27d': {
        'followup_prompt': "I have not settled on a course outline yet. Suggest three smartphone-only photography exercises for a beginner course, and name the composition principle each one teaches.",
        'reason': 'Removed the false claim that a nonexistent Module 2 had already been outlined; the existing three-exercise answer now directly fits.'},
    '57e24f8e7b54': {
        'followup_prompt': "Please create a science-fiction character named Elara. Give her an unusual way of perceiving the world and one central internal conflict.",
        'followup_answer': "Elara is a surveyor on a moon whose magnetic field shifts without warning. She perceives the world as music: metal structures hum in distinct pitches, and living creatures add faint rhythms. This lets her sense hidden tunnels and approaching storms, but the sounds grow painfully loud near crowded settlements. Her central conflict is whether to reveal that the moon's navigation beacon is destabilizing, even though exposing the truth could end the settlement she helped build. She must choose between protecting people's present security and giving them the chance to prepare for a dangerous future.",
        'category': 'completion',
        'reason': 'Replaced an invented reference to an absent Elara description with a self-contained character-writing request and answer.'},
    'ea7800b0393f': {
        'followup_prompt': "Use this fictional note: Elias, keeper of Gull Island lighthouse, collects starlight on clear nights with a silver ladle. What does he use to collect it?",
        'followup_answer': "He uses a silver ladle.",
        'reason': 'Turned a request to invent a passage and its question into a proper passage-grounded question.'},
    'ddfc8948fca2': {
        'followup_answer': "Three well-known U.S. campaign slogans are:\n\n1. **“I Like Ike” (Dwight D. Eisenhower, 1952):** The rhyme made the candidate's nickname easy to remember and repeat.\n2. **“Tippecanoe and Tyler Too” (William Henry Harrison and John Tyler, 1840):** Its rhythm and rhyme helped turn the ticket's names into a memorable chant.\n3. **“Yes We Can” (Barack Obama, 2008):** The short, collective phrase expressed optimism and invited supporters to see themselves as part of a shared effort.\n\nJohn F. Kennedy's “Ask not…” line is from his 1961 inaugural address, rather than a 1960 campaign slogan.",
        'reason': 'Replaced the misattributed Kennedy inaugural line and weak examples with accurate campaign-slogan examples.'},
}


def main():
    original = read_json(SOURCE)
    out = copy.deepcopy(original)
    rows = [r for split in ('train', 'dev', 'test') for r in out['data'][split]
            if r.get('source') == 'teacher']
    by_group = {}
    for r in rows:
        by_group.setdefault(r['group'], {})[r['kind']] = r
    applied = []
    missing = set(REPAIRS)
    for gid, spec in REPAIRS.items():
        match = next((g for k, g in by_group.items() if k.startswith(gid)), None)
        if match is None:
            continue
        missing.remove(gid)
        scenario = out['scenarios']['train']
        scenario = next(s for s in scenario if s['group'] == next(k for k, g in by_group.items() if g is match))
        if 'followup_prompt' in spec:
            follow = match['followup']
            follow['messages'][-2]['content'] = spec['followup_prompt']
            scenario['followup'] = spec['followup_prompt']
        if 'followup_answer' in spec:
            match['followup']['messages'][-1]['content'] = spec['followup_answer']
        if 'followup_answer_2' in spec:
            match['followup']['messages'][-1]['content'] = spec['followup_answer'] + '\n\n' + spec['followup_answer_2']
        if 'category' in spec:
            for r in match.values():
                r['category'] = spec['category']
            scenario['category'] = spec['category']
        if 'followup_answer' in spec or 'followup_answer_2' in spec:
            if scenario.get('reference_final') is not None:
                scenario['reference_final'] = match['followup']['messages'][-1]['content']
        applied.append(dict(group=next(k for k, g in by_group.items() if g is match),
                            repair=spec['reason']))
    if missing:
        raise ValueError('Expected audit example groups missing from training data: ' + ', '.join(sorted(missing)))

    # Recheck every teacher target for exact-count constraints; this previously
    # caught the two known one-paragraph / two-paragraph contradictions.
    failures = []
    checked = Counter()
    for r in rows:
        answer = r['messages'][-1]['content'].strip()
        prompt = r['messages'][-2]['content']
        for m in FORMAT.finditer(prompt):
            word = m.group(1).lower(); expected = NUMBERS[word] if word in NUMBERS else int(word)
            kind = 'paragraphs' if m.group(2).lower().startswith('paragraph') else 'bullets'
            actual = (len([x for x in re.split(r'\n\s*\n', answer) if x.strip()]) if kind == 'paragraphs'
                      else len(re.findall(r'^\s*[-*•]\s+', answer, re.M)))
            checked[kind] += 1
            if actual != expected:
                failures.append(dict(group=r['group'], kind=r['kind'], constraint=kind,
                                     expected=expected, actual=actual))
    if failures:
        raise ValueError('Unresolved exact formatting failures: ' + json.dumps(failures))

    out['data_hash'] = fingerprint(out['data'])
    out['curation'] = dict(method='manual corrections to documented defects; objective exact-format validation over every teacher target',
                           source_sha256=file_sha256(SOURCE), repairs=applied,
                           exact_format_checks=dict(checked), exact_format_failures=0,
                           note='The semantic audit previously sampled 32 groups plus targeted examples. This copy repairs every confirmed issue in that audit; other rows have not received a complete human semantic review.')
    OUT.mkdir(parents=True, exist_ok=True)
    atomic_json(OUT/'prepared.json', out)
    atomic_json(OUT/'repairs.json', out['curation'])
    lines = ['# Manually repaired teacher-chat data', '', out['curation']['note'], '',
             f"Source SHA-256: `{out['curation']['source_sha256']}`", '',
             f"Exact formatting checks: {dict(checked)}; failures: 0.", '', '## Repairs', '']
    for item in applied:
        lines.append(f"- `{item['group'][:12]}` — {item['repair']}")
    (OUT/'repairs.md').write_text('\n'.join(lines)+'\n')
    # The original source must remain byte-identical.
    assert file_sha256(SOURCE) == out['curation']['source_sha256']
    print(json.dumps(dict(source_sha256=file_sha256(SOURCE), repaired_data_hash=out['data_hash'],
                          manually_repaired_groups=len(applied), targets=2048,
                          exact_format_checks=dict(checked), exact_format_failures=0,
                          output=str(OUT/'prepared.json')), indent=2))


if __name__ == '__main__':
    main()
