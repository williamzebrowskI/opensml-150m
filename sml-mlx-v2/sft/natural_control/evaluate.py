"""Keep answers for blinded human review; no keyword scoring or automatic best."""
import json,random,re
from collections import Counter,defaultdict
from pathlib import Path
from sft.natural_control.engine import generate,nll
from sft.transfer_control.launch import prose_texts
from sft.transfer_control.engine import prose_loss
from sft.reading_repair.cards import records
DIR=Path(__file__).resolve().parent


def retention(split):
    groups=defaultdict(list)
    for row in records(split):groups[row['family']].append(row)
    result=[]
    for family,rows in sorted(groups.items()):
        result.extend(rows[:4])
    # Four per family, not the first 64 rows of a single family.
    return result


def inspect_structure(g):
    words=g['text'].lower().split();counts=Counter(tuple(words[i:i+4]) for i in range(len(words)-3))
    return dict(stopped=g['stop']=='eos',repeated=bool(counts and max(counts.values())>=4),empty=not bool(words))


def assess(b,data,split,cfg):
    probes=json.loads((DIR/'probes.json').read_text())[split];answers=[]
    for r in probes:
        g=generate(b,r['prompt'],cfg['max_new_tokens']);answers.append(dict(**r,generation=g,structure=inspect_structure(g)))
    natural=[]
    # Equal source coverage; reference answers are review aids, not perfect truth.
    for family in cfg['training_counts']:
        for r in [x for x in data[split] if x['family']==family][:4]:
            g=generate(b,r['prompt'],cfg['max_new_tokens']);natural.append(dict(id=r['id'],family=family,prompt=r['prompt'],source_answer=r['answer'],generation=g,structure=inspect_structure(g)))
    reading=[]
    for r in retention(split):
        g=generate(b,r['prompt'],64)
        def normalize(s):return ' '.join(re.findall('[a-z0-9]+',s.lower()))
        exact=g['stop']=='eos' and normalize(g['text']) in {normalize(x) for x in r['references']}
        reading.append(dict(id=r['id'],family=r['family'],prompt=r['prompt'],references=r['references'],generation=g,exact_diagnostic=exact))
    likelihood=nll(b,data[split],cfg);prose=prose_loss(b,prose_texts());b.model.eval()
    return dict(natural_likelihood=likelihood,prose_nll=prose,answers=answers,natural_answers=natural,reading_answers=reading,
        structure=dict(count=len(answers),stopped=sum(r['structure']['stopped'] for r in answers),repeated=sum(r['structure']['repeated'] for r in answers)),
        reading_exact_diagnostic=dict(correct=sum(r['exact_diagnostic'] for r in reading),count=len(reading),families=dict(Counter(r['family'] for r in reading))),
        selection='none; review actual correctness, faithfulness and completion; likelihood and exact match are diagnostics only')


def blind_packet(output):
    """Stable random candidate codes. Keep identity key separate from review text."""
    from sml_v2.common import atomic_json
    files=sorted(output.glob('*/evaluations/step_*.json'));rng=random.Random(25092631);rng.shuffle(files)
    packet=[];key={}
    for i,path in enumerate(files):
        code=f'C{i+1:03d}';key[code]=str(path.relative_to(output));d=json.loads(path.read_text())
        for kind in ('answers','natural_answers','reading_answers'):
            for row in d[kind]:
                packet.append(dict(candidate=code,kind=kind,id=row['id'],prompt=row['prompt'],answer=row['generation']['text'],
                    stop=row['generation']['stop'],rubric=row.get('rubric'),reference=row.get('reference',row.get('source_answer',row.get('references'))),
                    review=dict(correct=None,faithful=None,request_complete=None,notes='')))
    rng.shuffle(packet)
    atomic_json(output/'review_blinded.json',packet);atomic_json(output/'review_identity_key.json',key)
    return len(packet)
