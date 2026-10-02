"""A remains immutable; B must match its finished run except the UL term."""
from pathlib import Path
from sml_v2.common import read_json,file_sha256,fingerprint
from sft.conversation_ab import protocol as a
ROOT=a.ROOT;DIR=Path(__file__).resolve().parent
OUTPUT=ROOT/'runs/sft_conversation_ab_v1/B'
def code_files():return sorted(set(a.code_files()+list(DIR.glob('*.py'))))
def contract(cfg):
    base=read_json(a.DIR/'config.json')
    core={k:v for k,v in cfg.items() if k not in ('branch','unlikelihood')}
    if cfg['branch']!='B' or core!={k:v for k,v in base.items() if k!='branch'}:raise ValueError('B must match A settings')
    ul=cfg['unlikelihood']
    if ul!={'weight':0.1,'ngram':4,'max_new_tokens':96,'epsilon':1e-6,'generation':'greedy','scope':'non-rehearsal training prompts only'}:raise ValueError('Unexpected UL configuration')
    current=a.contract(base);completed=ROOT/'runs/sft_conversation_ab_v1/A'
    if read_json(completed/'report.json')['status']!='complete' or read_json(completed/'contract.json')!=current:raise ValueError('A is incomplete or common inputs changed')
    return dict(config=fingerprint(cfg),matched_A=current,code={str(p.relative_to(ROOT)):file_sha256(p) for p in code_files()},config_file=file_sha256(DIR/'config.json'))
def plan(cfg):
    p=a.plan(cfg);p.update(branch='B: supervised CE + sequence repetition unlikelihood',objective='Same CE as A plus 0.1 times repeated-token unlikelihood on own greedy continuations',unlikelihood=cfg['unlikelihood'],output=str(OUTPUT),comparison='Same 1920 parent, examples, order, optimizer, LR and 120 updates; extra generation increases compute')
    return p
