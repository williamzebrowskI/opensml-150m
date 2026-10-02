"""Frozen, resumable held-out conversation comparison: parent 768 vs moderate 1152."""
import argparse
import gc
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sml_v1.common import atomic_json, file_sha256, fingerprint, read_json, lock

WORK = ROOT / 'runs/teacher_chat_768_v1'
OUT = ROOT / 'diagnostics/teacher_chat_reserved_v1'
CONFIG = ROOT / 'sft/teacher_chat_768_v1/config.json'
DIMENSIONS = ('relevance', 'completion', 'history', 'support', 'instructions')
MODELS = {
    'parent-768': ('runs/sft_text_followup_512_v1/step_0000768_10f5ea2a3354',
                   '309ad719dc8569e482444b47b05f3fd24e02e7b7470dad2af132a976bf11ca02'),
    'teacher-1152': ('runs/teacher_chat_768_v1/moderate/step_0001152_2cdfc1d7d32b',
                     '3d18a96f34b44999102f7749f570c50c584d331b2a2b29d343e794215969396d'),
}


def prepare():
    cfg = read_json(CONFIG)
    prepared = read_json(WORK / 'data/prepared.json')
    assert fingerprint(prepared['data']) == prepared['data_hash']
    groups = {k: {r['group'] for r in rows} for k, rows in prepared['scenarios'].items()}
    assert len(groups['test']) == len(prepared['scenarios']['test']) == 32
    assert not groups['test'] & (groups['train'] | groups['dev'])
    assert not groups['test'] & {r.get('group') for r in prepared['data']['train']}
    paths = [Path(__file__), CONFIG, WORK / 'data/prepared.json',
             ROOT / 'runs/stage_b_prose_v1/step_0073243_7f3070237eea/model.safetensors.json']
    for folder in ('sft/teacher_chat_768_v1', 'sft/conversation_foundation_original',
                   'sft/intact_smoltalk_base_pilot', 'sft/skill_balance', 'sml_v1'):
        paths.extend((ROOT / folder).glob('*.py'))
    for name, (bundle, expected) in MODELS.items():
        b = ROOT / bundle
        manifest = read_json(b / 'manifest.json')
        assert file_sha256(b / 'model.safetensors') == expected
        assert manifest['files']['model.safetensors']['sha256'] == expected
        assert file_sha256(b / 'model.safetensors.json') == manifest['files']['model.safetensors.json']['sha256']
        paths.extend([b / 'model.safetensors', b / 'model.safetensors.json', b / 'manifest.json'])
    frozen = dict(models=MODELS, split='test', conversations=32, turns_per_model=64,
                  max_new_tokens=cfg['max_new_tokens'], greedy=True, own_history=True,
                  training=False, automatic_selection=False,
                  protected={str(p): file_sha256(p) for p in paths})
    path = OUT / 'manifest.json'
    if path.exists() and read_json(path) != json.loads(json.dumps(frozen)):
        raise ValueError('Reserved comparison inputs changed')
    return cfg, prepared['scenarios']['test'], frozen


def answers(cfg, scenarios, name, frozen):
    import mlx.core as mx
    from sft.skill_balance.engine import load
    from sft.conversation_foundation_original.engine import generate
    path = OUT / (name + '.answers.json')
    saved = read_json(path) if path.exists() else dict(identity=fingerprint(frozen), fresh_answers=[])
    assert saved['identity'] == fingerprint(frozen)
    assert [r['group'] for r in saved['fresh_answers']] == [r['group'] for r in scenarios[:len(saved['fresh_answers'])]]
    if len(saved['fresh_answers']) == len(scenarios):
        return saved
    b = load(cfg, weights=ROOT / MODELS[name][0] / 'model.safetensors')
    try:
        for row in scenarios[len(saved['fresh_answers']):]:
            history = list(row['prefix'])
            turns = []
            for i in range(2):
                if i:
                    history.append(dict(role='user', content=row['followup']))
                result = generate(b, history, cfg['max_new_tokens'])
                turns.append(dict(history=list(history), **result))
                history.append(dict(role='assistant', content=result['text']))
            saved['fresh_answers'].append(dict(group=row['group'], category=row['category'], turns=turns))
            atomic_json(path, saved)
            print(f'[reserved-answers] {name} {len(saved["fresh_answers"])}/32', flush=True)
    finally:
        del b
        gc.collect()
        mx.clear_cache()
    return saved


def judge(cfg, evaluations):
    from sft.teacher_chat_768_v1.teacher import Teacher, JUDGE
    teacher = Teacher(cfg)
    def validate(obj):
        if any(type(obj.get(k)) is not int or not 1 <= obj[k] <= 5 for k in DIMENSIONS):
            raise ValueError('Invalid score schema')
    records = []
    # Alternate model order by conversation, without exposing model identities to judge.
    try:
        for index in range(32):
            names = list(MODELS)
            if index % 2:
                names.reverse()
            for name in names:
                answer = evaluations[name]['fresh_answers'][index]
                for ti, turn in enumerate(answer['turns']):
                    key = fingerprint(dict(prompt=turn['history'], text=turn['text'], judge=JUDGE,
                                           teacher=teacher.identity))
                    path = OUT / 'judge_cache' / (key + '.json')
                    if path.exists():
                        record = read_json(path)
                    else:
                        try:
                            obj = teacher.call(JUDGE, dict(history=turn['history'], response=turn['text']),
                                               max_tokens=500, validator=validate)
                            record = dict(status='scored', scores=obj,
                                          mean=sum(obj[k] for k in DIMENSIONS) / len(DIMENSIONS))
                        except (ValueError, json.JSONDecodeError) as exc:
                            record = dict(status='unscored', reason=str(exc))
                        atomic_json(path, record)
                    records.append(dict(model=name, group=answer['group'], category=answer['category'],
                                        turn=ti + 1, **record))
                    print(f'[reserved-judge] {len(records)}/128; {name}; {record.get("mean", "unscored")}', flush=True)
    finally:
        teacher.close()
    atomic_json(OUT / 'judgments.json', records)
    return records


def report(evaluations, records):
    from sft.conversation_foundation_original.engine import repeated
    def scores(rows):
        valid = [r for r in rows if r['status'] == 'scored']
        return dict(valid=len(valid), expected=len(rows),
                    mean=sum(r['mean'] for r in valid)/len(valid) if valid else None,
                    dimensions={k:sum(r['scores'][k] for r in valid)/len(valid) if valid else None for k in DIMENSIONS})
    result = dict(training=False, split='reserved test', models={},
                  note='Same-teacher diagnostic, not independent or official MT-Bench. Follow-ups were authored using frozen parent drafts. This test set is now evaluated; do not reuse it for iterative tuning.')
    for name in MODELS:
        rows = [r for r in records if r['model'] == name]
        ts = [t for a in evaluations[name]['fresh_answers'] for t in a['turns']]
        result['models'][name] = dict(overall=scores(rows),
            by_turn={str(i):scores([r for r in rows if r['turn']==i]) for i in (1,2)},
            by_category={k:scores([r for r in rows if r['category']==k]) for k in sorted({r['category'] for r in rows})},
            stopped=sum(t['stop']=='eos' for t in ts)/len(ts),
            repeated=sum(repeated(t['tokens']) for t in ts)/len(ts),
            mean_tokens=sum(len(t['tokens']) for t in ts)/len(ts),
            stop_counts=dict(Counter(t['stop'] for t in ts)))
    indexed={(r['model'],r['group'],r['turn']):r for r in records}
    deltas=[]
    counts=Counter()
    conversation_deltas=[]
    for a in evaluations['parent-768']['fresh_answers']:
        ds=[]
        for ti in (1,2):
            p=indexed[('parent-768',a['group'],ti)];c=indexed[('teacher-1152',a['group'],ti)]
            if p['status']!='scored' or c['status']!='scored':
                counts['unscored_pairs']+=1
                continue
            delta=c['mean']-p['mean'];ds.append(delta);deltas.append(delta)
            counts['wins' if delta>1e-9 else 'losses' if delta<-1e-9 else 'ties']+=1
        if len(ds)==2:conversation_deltas.append(sum(ds)/2)
    result['paired_turns']=dict(counts)
    if len(conversation_deltas)==32:
        rng=random.Random(9292026)
        boot=sorted(sum(rng.choices(conversation_deltas,k=32))/32 for _ in range(10000))
        result['paired_conversation_mean_delta']=sum(conversation_deltas)/32
        result['paired_conversation_bootstrap_95_interval']=[boot[249],boot[9749]]
    atomic_json(OUT / 'summary.json', result)
    lines=['# Reserved conversation comparison','',result['note'],'',
           '| Metric | Parent 768 | Teacher 1152 |','|---|---:|---:|']
    for metric in ('relevance','completion','history','support','instructions'):
        vals=[result['models'][n]['overall']['dimensions'][metric] for n in MODELS]
        lines.append(f'| {metric} | {vals[0]:.3f} | {vals[1]:.3f} |')
    lines+=['', '```json',json.dumps(result,indent=2),'```']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')
    lines=['# Side-by-side reserved transcripts','']
    for i, a in enumerate(evaluations['parent-768']['fresh_answers']):
        lines += [f'## {i+1}. {a["category"]} / {a["group"][:12]}','']
        for ti in (0,1):
            lines += [f'### Turn {ti+1}','', '**Request**', a['turns'][ti]['history'][-1]['content'],'']
            for name in MODELS:
                t=evaluations[name]['fresh_answers'][i]['turns'][ti]
                r=indexed[(name,a['group'],ti+1)]
                lines += ['**'+name+'**',t['text'],'',f'Stop: {t["stop"]}; score: {r.get("mean")}',str(r.get('scores',{}).get('reason','')),'']
    (OUT/'transcripts.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(result,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true')
    args=parser.parse_args()
    cfg, scenarios, frozen=prepare()
    print('[checked] 32 disjoint reserved conversations; both checkpoint hashes verified',flush=True)
    if not args.run:return
    with lock(ROOT/'sft/.experiment.lock'):
        atomic_json(OUT/'manifest.json',frozen)
        evaluations={name:answers(cfg,scenarios,name,frozen) for name in MODELS}
        records=judge(cfg,evaluations)
        report(evaluations,records)
        for path,sha in frozen['protected'].items():
            if file_sha256(path)!=sha:raise ValueError('Input changed: '+path)
        atomic_json(OUT/'integrity.json',dict(inputs_unchanged=True,training=False))
        print('[finished]',OUT/'report.md',flush=True)


if __name__=='__main__':main()
