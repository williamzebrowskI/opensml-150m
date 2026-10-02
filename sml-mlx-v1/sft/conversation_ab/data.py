"""Pinned OASST2 in bounded RAM; reviewed IDs, never a raw corpus disk cache."""
import hashlib, io, random, re
from pathlib import Path
from collections import Counter
from sml_v1.common import read_json, fingerprint
ROOT=Path(__file__).resolve().parents[2]
DIR=Path(__file__).resolve().parent
SOURCE=dict(repo='OpenAssistant/oasst2',revision='179dd21fc55192153d94adb0e0ce8f69e222bf75',file='data/train-00000-of-00001-88ba0162028a73fc.parquet',bytes=63494018,sha256='9638227b41b0a4fb69afa5d5c4b52e3fb6abfd56c9870d81bae32c9981e769b1',license='Apache-2.0')
BLOCK=re.compile(r'```|https?://|<\||\b(python|javascript|typescript|programming|coding|code|algorithm|algebra|calculus|equation|arithmetic|mathematics|calculate|calculation|multiply|multiplication|division|percent|formula|sql|linux|software|computer|terminal|chatgpt|openassistant|language model|AI assistant|as an ai|diagnosis|medication|dosage|suicide|self.harm|investment|stock market|lawsuit|legal advice|sex|sexual|rape|weapon|bomb|kill|religion|religious|politic|political)\b|\d\s*[+*/=]\s*\d',re.I)
ROLE=re.compile(r'(^|\n)\s*(User|Assistant|System|Human):',re.I)
def norm(s):return ' '.join(re.findall(r'[a-z0-9]+',s.lower()))
def split(group):
    n=int(fingerprint(['conversation-ab',group])[:8],16)%10
    return 'train' if n<8 else 'dev' if n==8 else 'test'
def stream(cancelled=lambda:False):
    import requests,pyarrow.parquet as pq
    buf=io.BytesIO();h=hashlib.sha256();n=0
    url=f"https://huggingface.co/datasets/{SOURCE['repo']}/resolve/{SOURCE['revision']}/{SOURCE['file']}"
    with requests.get(url,stream=True,timeout=(20,90)) as r:
        r.raise_for_status()
        for chunk in r.iter_content(65536):
            if cancelled():raise InterruptedError('Preparation stopped')
            n+=len(chunk)
            if n>SOURCE['bytes']:raise ValueError('Source exceeded bound')
            h.update(chunk);buf.write(chunk)
    if n!=SOURCE['bytes'] or h.hexdigest()!=SOURCE['sha256']:raise ValueError('Source integrity failure')
    buf.seek(0)
    # The immutable 63.5 MB file is decoded in batches; keep only English fields
    # required to reconstruct ancestry, then free the compressed bytes.
    rows={}
    for batch in pq.ParquetFile(buf).iter_batches(batch_size=256):
        if cancelled():raise InterruptedError('Preparation stopped')
        for r in batch.to_pylist():
            if r['lang']=='en': rows[r['message_id']]={k:r[k] for k in ('message_id','parent_id','text','role','review_result','deleted','rank','synthetic','message_tree_id','labels')}
    buf.close();return rows

def candidates(tok,cancelled=lambda:False):
    rows=stream(cancelled);accepted=[];rejected=Counter()
    from sft.skill_balance.data import exclusions,grams
    denied,dgrams,_=exclusions()
    for name in ('generation_comparison_1920_v1','continuation_audit_1920_v1'):
        p=ROOT/'diagnostics'/name/'responses.jsonl'
        if p.exists():
            import json
            for line in p.read_text().splitlines():
                r=json.loads(line);text=r.get('prompt',r.get('input',''))
                denied.add(norm(text));dgrams |= grams(text)
    for r in rows.values():
        if r['role']!='assistant' or r['rank']!=0 or r['synthetic'] or r['deleted'] or not r['review_result']:continue
        labs=dict(zip((r['labels'] or {}).get('name',[]),(r['labels'] or {}).get('value',[])))
        if labs.get('quality',0)<.75 or labs.get('helpfulness',0)<.75 or labs.get('fails_task',0)>0:continue
        a=r['text'].strip()
        if not 12<=len(a.split())<=110 or not a.endswith(('.', '!', '?', '"')):continue
        chain=[];p=r['parent_id'];visited=set()
        while p:
            if p in visited or p not in rows:chain=[];break
            visited.add(p);parent=rows[p];chain.append(parent);p=parent['parent_id']
            if len(chain)>3:chain=[];break
        chain=chain[::-1]
        if len(chain) not in (1,3) or [x['role'] for x in chain] != (['prompter'] if len(chain)==1 else ['prompter','assistant','prompter']):continue
        if any(x['deleted'] or x['synthetic'] or not x['review_result'] for x in chain):continue
        if len(chain)==3 and chain[1]['rank']!=0:continue
        texts=[x['text'].strip() for x in chain]+[a]
        if any(BLOCK.search(t) or ROLE.search(t) for t in texts):continue
        if sum(len(t.split()) for t in texts)>300:continue
        if any(norm(t) in denied or grams(t)&dgrams for t in texts):continue
        if len(chain)==1:prompt=texts[0]
        else:prompt='Previous messages:\nUser: '+texts[0]+'\nAssistant: '+texts[1]+'\n\nCurrent message: '+texts[2]
        head=tok.encode('User: '+prompt+'\nAssistant:');ids=tok.encode('User: '+prompt+'\nAssistant: '+a)
        if ids[:len(head)]!=head or len(ids)>1024:continue
        group=r['message_tree_id'];accepted.append(dict(id=r['message_id'],group=group,split=split(group),prompt=prompt,answer=a,source='oasst2',turns=len(chain),quality=labs.get('quality'),helpfulness=labs.get('helpfulness')))
    accepted.sort(key=lambda r:fingerprint(['review-order',r['id']]))
    print('[source] eligible OASST2:',dict(Counter(r['split'] for r in accepted)),flush=True)
    return accepted

def rebuild(tok,cancelled=lambda:False):
    sel=read_json(DIR/'selection.json');wanted={r['id']:r for r in sel['oasst']}
    source={r['id']:r for r in candidates(tok,cancelled) if r['id'] in wanted}
    if set(source)!=set(wanted):raise ValueError('Reviewed source IDs missing')
    for key,row in source.items():
        if fingerprint(row)!=wanted[key]['hash']:raise ValueError('Reviewed response changed: '+key)
    originals=read_json(DIR/'originals.json');data={s:[source[x['id']] for x in sel['oasst'] if x['split']==s] for s in ('train','dev','test')}
    for s in data:data[s]+=originals[s]
    # Exact previous source selection; only examples consumed before 1920.
    from sft.skill_balance.data import build
    old,manifest=build(read_json(ROOT/'sft/skill_balance/config.json'),cancelled=cancelled)
    if manifest!=read_json(ROOT/'sft/skill_balance/selection.json'):raise ValueError('Prior rehearsal selection changed')
    pool={r['id']:r for f,n in (('commonsense',4096),('instruction',4096),('reading',2048)) for r in old['train'][f][:n]}
    rehearsal=[]
    for key in sel['rehearsal_ids']:
        if key not in pool:raise ValueError('Rehearsal was not consumed by parent')
        rehearsal.append(dict(pool[key],source='rehearsal'))
    data['retention']={f:old['dev'][f][:32] for f in ('commonsense','instruction','reading')}
    data['retention']['commonsense']=[r for name in ('commonsenseqa','socialiqa') for r in [x for x in old['dev']['commonsense'] if x['source']==name][:16]]
    humans=[r for r in data['train'] if r['source']=='oasst2'];new=[r for r in data['train'] if r['source']=='original']
    rng=random.Random(sel['seed'])
    for group in (humans,new,rehearsal):rng.shuffle(group)
    # Exact 60/15/25 counts in each 20-example block, then interleave and split
    # into a frozen pool. The engine shuffles this pool once per epoch.
    train=[]
    for i in range(len(humans)//12):
        block=humans[i*12:(i+1)*12]+new[i*3:(i+1)*3]+rehearsal[i*5:(i+1)*5];rng.shuffle(block);train+=block
    if len(train)!=sel['train_count'] or len({r['id'] for r in train})!=len(train):raise ValueError('Exposure count/duplicate ID')
    data['train']=train
    for s in ('train','dev','test'):
        if fingerprint(data[s])!=sel['hashes'][s]:raise ValueError('Frozen selection differs: '+s)
    return data
