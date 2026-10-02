"""Trial21's broader public sources, with fixed controls and longer exposure."""
import copy
import random
import shutil
from collections import Counter
from pathlib import Path
import data_focused
from sml_v1.common import atomic_json, read_json
from sml_v1.tokenization import Tokenizer

ROOT = Path(__file__).resolve().parents[3]
PUBLIC = ROOT / 'experiments/public_repair_384_v1'
rules = data_focused.rules


def prepare(c, cfg):
    assert cfg['updates'] == 512
    stage = c / 'prepare_staging'
    (stage / 'raw').mkdir(parents=True)
    pins = read_json(c / 'source_pins.json')
    for key, pin in pins.items():
        shutil.copyfile(c / 'raw' / pin['local_file'], stage / 'raw' / (key+'.parquet'))
    atomic_json(stage / 'source_pins.json', pins)
    try:
        subcfg = copy.deepcopy(cfg)
        subcfg.update(updates=256, ul_weight=0., kl_weight=0.)
        data_focused.prepare(stage, subcfg)
        original = read_json(stage / 'data.json')
        selection = read_json(stage / 'selection.json')
    finally:
        shutil.rmtree(stage)
    fresh = {k:[r for r in original['train'] if r.get('dataset_category')==k]
             for k in ('constraints','squad','sciq')}
    assert {k:len(v) for k,v in fresh.items()} == dict(constraints=1536,squad=1024,sciq=512)
    raw = read_json(PUBLIC / 'data/prepared.json')['data']
    rng = random.Random(cfg['seed'])
    replay = {s:[dict(r,parent_replay=True) for r in raw['train'][:2048] if r['source']==s]
              for s in ('chat','human','grounded','instructions')}
    for rows in replay.values():
        assert len(rows)==512
        rng.shuffle(rows)
    train = []
    quotas = dict(constraints=6,squad=4,sciq=2)
    for epoch in range(2):
        pools = {k:list(v) for k,v in fresh.items()}
        for rows in pools.values():
            rng.shuffle(rows)
        for u in range(256):
            index = epoch*256+u
            batch = [dict(r,parent_replay=False) for k,count in quotas.items()
                     for r in pools[k][u*count:(u+1)*count]]
            batch += [rows[index] for rows in replay.values()]
            assert len(batch)==16
            rng.shuffle(batch)
            train.extend(batch)
    assert len(train)==8192 and len({r['group'] for r in train})==5120
    held = original['targeted_dev']+original['targeted_test']+raw['dev']+raw['test']
    assert not {rules.group_id(r['messages']) for r in train} & {rules.group_id(r['messages']) for r in held}
    tok = Tokenizer(ROOT / 'tokenizer/bytebpe32k_v1')
    encoded = [e for r in train for e in rules.turns(tok,r,cfg['context'])]
    assert all(e['y'][-1]==tok.eos and len(e['x'])==len(e['y'])<=cfg['context'] for e in encoded)
    cfg.update(data_focused=False, dataset_plan='breadth-stable',
               evaluation_updates=[0,128,256,384,512], checkpoint_keep=4,
               anchor_replay_only=True, negative_replay_only=True,
               training_counts=dict(Counter(r['source'] for r in train)),
               method='Broader verified public instruction/annotated QA mixture; two fresh passes plus repair512 replay; assistant-only CE + EOS; repetition penalty .3 and replay KL anchor 2')
    atomic_json(c/'config.json',cfg)
    atomic_json(c/'data.json',dict(dataset_plan='breadth-stable',train=train,dev=raw['dev'],test=raw['test'],
                                 targeted_dev=original['targeted_dev'],targeted_test=original['targeted_test']))
    selection.update(training_conversations=8192,unique_conversations=5120,
                     fresh_passes=2,parent_replay_fraction=.25,
                     dataset_mix=dict(smol_constraints=3072,squad=2048,sciq=1024,parent_replay=2048),
                     assistant_turns=len(encoded),stability_controls=dict(ul=.3,kl=2,anchor_replay_only=True,negative_replay_only=True),
                     maximum_updates=512,development_check_updates=[128,256,384,512],
                     interpretation='Reuses trial21 source selection and broad format coverage. Longer exposure plus fixed controls are tested together; causal contribution is not isolated.')
    atomic_json(c/'selection.json',selection)
    print('[prepared-broad-stable]',selection,flush=True)
