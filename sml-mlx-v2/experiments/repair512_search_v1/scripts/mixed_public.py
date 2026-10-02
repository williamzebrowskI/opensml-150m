"""Balanced public TRAIN sources with fixed repetition and replay protections."""
import copy
import random
import shutil
from collections import Counter
from pathlib import Path

from sml_v2.common import atomic_json, read_json
from sml_v2.tokenization import Tokenizer
import dataset_trials
import short_constraints
import human_dialogue
import science_dialogue

ROOT = Path(__file__).resolve().parents[3]
PUBLIC = ROOT / 'experiments/public_repair_384_v1'
rules = dataset_trials.rules


def prepare(c, cfg):
    assert cfg['updates'] == 256
    pins = read_json(c / 'source_pins.json')
    builders = [
        ('constraints', 'constraints-replay', short_constraints),
        ('squad', 'squad-replay', dataset_trials),
        ('oasst', 'human-replay', human_dialogue),
        ('openbook', 'science-replay', science_dialogue),
    ]
    fresh, dev, test, receipts = {}, [], [], {}
    staging = c / 'prepare_staging'
    staging.mkdir()
    try:
        for category, plan, builder in builders:
            stage = staging / category
            (stage / 'raw').mkdir(parents=True)
            pin = pins[category]
            shutil.copyfile(c / 'raw' / pin['local_file'], stage / 'raw' / pin['local_file'])
            atomic_json(stage / 'source_pins.json', {category: pin})
            subcfg = copy.deepcopy(cfg)
            subcfg.update(dataset_plan=plan, updates=128, fresh_per_batch=4,
                          ul_weight=0., kl_weight=0.)
            builder.prepare(stage, subcfg)
            data = read_json(stage / 'data.json')
            fresh[category] = [r for r in data['train'] if r.get('dataset_category') == category]
            assert len(fresh[category]) == 512
            dev.extend(data['targeted_dev'])
            test.extend(data['targeted_test'])
            receipts[category] = read_json(stage / 'selection.json')
    finally:
        shutil.rmtree(staging)
    raw = read_json(PUBLIC / 'data/prepared.json')['data']
    rng = random.Random(cfg['seed'])
    replay = {s: [dict(r, parent_replay=True) for r in raw['train'][:2048]
                  if r['source'] == s] for s in ('chat', 'human', 'grounded', 'instructions')}
    for rows in replay.values():
        assert len(rows) == 512
        rng.shuffle(rows)
    train = []
    for u in range(256):
        batch = [dict(r, parent_replay=False) for rows in fresh.values()
                 for r in rows[2*u:2*u+2]]
        batch += [r for rows in replay.values() for r in rows[2*u:2*u+2]]
        rng.shuffle(batch)
        assert len(batch) == 16
        train.extend(batch)
    assert len(train) == len({r['group'] for r in train}) == 4096
    train_groups = {rules.group_id(r['messages']) for r in train}
    held_groups = {rules.group_id(r['messages']) for r in dev + test + raw['dev'] + raw['test']}
    assert not train_groups & held_groups
    assert not {r['group'] for r in dev} & {r['group'] for r in test}
    tok = Tokenizer(ROOT / 'tokenizer/bytebpe32k_v1')
    encoded = [e for r in train for e in rules.turns(tok, r, cfg['context'])]
    assert all(e['y'][-1] == tok.eos and len(e['x']) == len(e['y']) <= cfg['context'] for e in encoded)
    cfg.update(evaluation_updates=[0,128,256], anchor_replay_only=True,
               training_counts=dict(Counter(r['source'] for r in train)),
               method='Balanced public TRAIN mixture + 50% repair512 replay; assistant-only CE + EOS; reference-aware on-policy repetition penalty .3 and replay KL anchor 2')
    atomic_json(c / 'config.json', cfg)
    atomic_json(c / 'data.json', dict(dataset_plan='mixed-public', train=train,
                                    dev=raw['dev'], test=raw['test'], targeted_dev=dev, targeted_test=test))
    receipt = dict(training_conversations=4096, unique_conversations=4096,
                   fresh_public_examples=2048, parent_replay_fraction=.5,
                   dataset_mix={**{k:512 for k in fresh}, 'parent_replay':2048},
                   assistant_turns=len(encoded), fresh_dev=len(dev), fresh_reserved_test=len(test),
                   source_selections=receipts, whole_answers=True, targets_truncated=False,
                   benchmark_examples_used_for_training=False, replay_anchor_only=True,
                   limitations='Source annotations and named checks do not guarantee semantic correctness. Multiple data sources plus previously tested stability controls change together; this trial does not isolate a causal contribution.')
    atomic_json(c / 'selection.json', receipt)
    print('[prepared-balanced-public]', {k:v for k,v in receipt.items() if k!='source_selections'}, flush=True)


def assess(c, b, dataset, u):
    from sft.conversation_foundation_v1.engine import generate, repeated
    from data_focused import verify
    p = c / 'runs/sft/evaluations' / f'targeted_{u:05d}.json'
    if p.exists():
        return
    answers, metrics = [], {}
    for category in ('constraints', 'squad', 'oasst', 'openbook'):
        rows = [r for r in dataset['targeted_dev'] if r['dataset_category']==category][:16]
        current = []
        for row in rows:
            messages = []
            for m in row['messages']:
                if m['role']=='user':
                    messages.append(m)
                    continue
                g = generate(b, messages, 384)
                item = dict(id=row['id'], category=category, prompt=list(messages),
                            reference=m['content'], repeated=repeated(g['tokens']), **g)
                if category=='constraints':
                    item['named_constraint_pass']=all(verify(g['text'],row['named_checks']))
                elif category=='squad':
                    item['exact_match']=any(dataset_trials.normal(g['text'])==dataset_trials.normal(a) for a in row['references'])
                    item['token_f1']=max(dataset_trials.f1(g['text'],a) for a in row['references'])
                elif category=='openbook':
                    item['gold_mention_proxy']=' '.join(dataset_trials.normal(row['gold_answer'])) in ' '.join(dataset_trials.normal(g['text']))
                current.append(item)
                messages.append(dict(role='assistant', content=g['text']))
        n = len(current)
        out = dict(generated_turns=n, stopped=sum(a['stop']=='eos' for a in current)/n,
                   repeated=sum(a['repeated'] for a in current)/n,
                   empty=sum(not a['text'].strip() for a in current)/n)
        for key in ('named_constraint_pass','exact_match','token_f1','gold_mention_proxy'):
            if key in current[0]:
                out[key]=sum(a[key] for a in current)/n
        metrics[category]=out
        answers.extend(current)
        print('[fresh-balanced-public-eval]',u,category,out,flush=True)
    atomic_json(p,dict(metrics=metrics,answers=answers,reserved_test_used=False))
