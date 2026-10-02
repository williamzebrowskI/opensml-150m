"""One bounded source pass reconstructs parent replay and selects unseen replies."""
import heapq,random,re
from collections import Counter,defaultdict
from pathlib import Path
from sml_v1.common import fingerprint,read_json
from sml_v1.tokenization import Tokenizer
from sft.base_curriculum_v1 import data as parent
from sft.natural_control import data as old
from .candidate import candidate
ROOT=Path(__file__).resolve().parents[2];DIR=Path(__file__).resolve().parent
FILLER=re.compile(r"\b(?:I(?:'m| am) (?:ready|here|happy|glad|excited) to (?:help|assist)|I(?:'d| would) (?:love|be happy) to help|let(?:'s| us) (?:start|begin) by|what(?:'s| is) your (?:budget|research question))\b",re.I)
BAD_FACT=re.compile(r'\b(?:photosynthesis releases carbon dioxide|sun revolves around the earth)\b',re.I)
EXTRA_SCOPE=re.compile(r'\b(?:portray|impersonate|can you be|you indicated|this session|debug\w*|code|roleplay|role.play|wait for my response|round [0-9]|cut.?off date|you mentioned|Wilson.s disease|Kayser.Fleischer|sleepwalking|sleep apnea|cholangi\w*|cirrhosis)\b',re.I)
GREETING=re.compile(r'^(?:hi|hello|hey|good (?:morning|evening|afternoon)|thanks|thank you)[!., ]*$',re.I)

def quality(row):
    if not parent.quality(row):return False
    if old.LIVE.search(row['prompt']+' '+row['answer']) or old.MEDICAL.search(row['prompt']+' '+row['answer']):return False
    if re.search(r'\b(?:current wind|wind speed outside|what.s the weather|weather forecast)\b',row['prompt'],re.I):return False
    if EXTRA_SCOPE.search(row['prompt']+' '+row['answer']) or re.search(r'^(?:be a|be an)\b',row['prompt'],re.I):return False
    answer=row['answer'].replace('’',"'")
    if FILLER.search(answer) or BAD_FACT.search(answer):return False
    if old.norm(row['prompt'])==old.norm(answer):return False
    if row['family'] in ('rewrite','summary') and re.search(r"\b(?:I(?:'m| am| will)|let me|would you|could you|please provide|please share)\b",answer,re.I):return False
    if row['family']=='explanation' and answer.endswith('?'):return False
    return True

def from_raw(file,index,raw,tok,cfg):
    if raw['source'] not in old.SOURCES:return
    msgs=raw['messages'];system=[]
    if msgs and msgs[0]['role']=='system':system=msgs[:1];msgs=msgs[1:]
    if len(msgs)<2 or msgs[0]['role']!='user':return
    first=next((m['content'] for m in msgs if m['role']=='user' and len(m['content'].split())>=4),msgs[0]['content'])
    group=fingerprint(old.norm(first))
    for j in range(0,min(len(msgs)-1,8),2):
        if [m['role'] for m in msgs[:j+2]]!=['user','assistant']*((j+2)//2):continue
        if j and raw['source'] not in ('smol-magpie-ultra-short','everyday-conversations'):continue
        row,_=candidate(dict(raw,messages=system+msgs[j:j+2]),cfg,tok,tok.encode)
        if row is None:continue
        if j:
            history='\n'.join(('User: ' if k%2==0 else 'Assistant: ')+m['content'].strip() for k,m in enumerate(msgs[:j]))
            if len(history.split())>200 or old.BLOCK.search(history) or old.MATH.search(history) or old.PERSONA.search(history) or old.BAD_RESPONSE.search(history):continue
            row['prompt']='Previous messages:\n'+history+'\n\nCurrent message: '+row['prompt'];row['family']='followup'
        elif row['family']=='conversation':
            if parent.CREATIVE.search(row['prompt']):continue
            if parent.EXPLAIN.search(row['prompt']):row['family']='explanation'
        row.update(id=fingerprint([raw['source'],row['prompt'],row['answer']]),group=group,split=parent.partition(group),source=old.REPO,source_file=file,source_row=index,turn=j//2)
        if row['family'] not in cfg['new_counts']:continue
        if parent.valid(tok,row,cfg) and quality(row):yield row

_CACHE=None

def build(cfg,cancelled=lambda:False,review=False):
    global _CACHE
    if _CACHE is None:
        tok=Tokenizer(ROOT/'tokenizer/bytebpe32k_v1');heaps=defaultdict(list);counts=Counter();serial=0
        def accept(row):
            nonlocal serial
            if row['split']!='train':return
            counts[row['family']]+=1;serial+=1
            entry=(-int(fingerprint([cfg['seed'],row['id']]),16),row['id'],serial,row)
            h=heaps[row['family']]
            if len(h)<8192:heapq.heappush(h,entry)
            elif entry[:2]>h[0][:2]:heapq.heapreplace(h,entry)
        original=old.stream
        def capture(cancelled=lambda:False):
            for file,index,raw in original(cancelled):
                for row in from_raw(file,index,raw,tok,cfg):accept(row)
                yield file,index,raw
        parent._COLLECTION=None;old.stream=capture
        try:previous,pm=parent.build(read_json(ROOT/'sft/base_curriculum_v1/config.json'),cancelled)
        finally:old.stream=original
        if fingerprint(pm)!=fingerprint(read_json(ROOT/'sft/base_curriculum_v1/selection.json')):raise ValueError('Parent selections failed reproduction')
        for entries in parent._COLLECTION[1].values():
            for *_,row in entries:
                if row['family'] in ('uncertainty','instruction') and quality(row):accept(row)
        parent._COLLECTION=None
        _CACHE=(previous,heaps,counts,pm)
    previous,heaps,counts,pm=_CACHE
    denied,dgrams=old.exclusions()
    from sft.skill_balance.data import exclusions
    from sft.transfer_control.launch import prose_texts
    extra,xgrams,_=exclusions();denied|=extra;dgrams|=xgrams
    for rows in read_json(ROOT/'sft/base_restart/examples.json').values():
        for r in rows:denied.add(old.norm(r['prompt']));dgrams|=old.grams(r['prompt'])
    manual=read_json(DIR/'review_prompts.json');reserved=read_json(DIR/'reserved_prompts.json')
    for r in manual+reserved+previous['manual_review']:
        denied.add(old.norm(r['prompt']));dgrams|=old.grams(r['prompt'])
    for s in prose_texts():denied.add(old.norm(s));dgrams|=old.grams(s)
    all_groups={r['group'] for split in ('train','dev','test') for r in previous[split]}
    for split in ('dev','test'):
        for r in previous[split]:denied.add(old.norm(r['prompt']));dgrams|=old.grams(r.get('content',r['prompt']))
    excluded=set(cfg['excluded_ids']);seen=set();groups=set();new=[];replay=[]
    def allowed(r):
        return r['id'] not in excluded and old.norm(r['prompt']) not in denied and not old.grams(r.get('content',r['prompt']))&dgrams and old.norm(r['prompt']) not in seen and r['group'] not in groups and quality(r)
    available={}
    for family,count in cfg['new_counts'].items():
        rows=[]
        for *_,r in sorted(heaps[family],key=lambda e:e[:2],reverse=True):
            if r['group'] in all_groups or not allowed(r):continue
            rows.append(dict(r,role='new'));seen.add(old.norm(r['prompt']));groups.add(r['group'])
        available[family]=len(rows)
        if len(rows)<count:raise ValueError(f'Insufficient NEW {family}: {len(rows)}/{count}; scan={dict(counts)}')
        # Release unselected groups so other families can use them.
        for r in rows[count:]:seen.remove(old.norm(r['prompt']));groups.remove(r['group'])
        new+=rows[:count]
    for family,count in cfg['rehearsal_counts'].items():
        rows=sorted((r for r in previous['train'] if r['family']==family),key=lambda r:fingerprint([cfg['seed'],'replay',r['id']]))
        chosen=[]
        for r in rows:
            if not allowed(r):continue
            chosen.append(dict(r,role='rehearsal'));seen.add(old.norm(r['prompt']));groups.add(r['group'])
            if len(chosen)==count:break
        if len(chosen)!=count:raise ValueError(f'Insufficient REHEARSAL {family}: {len(chosen)}/{count}')
        replay+=chosen
    # Fixed 8 new + 8 rehearsal per optimizer update, without replacement.
    rng=random.Random(cfg['seed']);rng.shuffle(new);rng.shuffle(replay);ordered=[]
    for j in range(cfg['updates']):
        batch=new[j*8:j*8+8]+replay[j*8:j*8+8];rng.shuffle(batch);ordered+=batch
    data=dict(train=ordered,dev=previous['dev'],test=previous['test'],manual_review=manual,reserved_review=reserved)
    data['train_probes']=[r for family in cfg['training_counts'] for r in [x for x in ordered if x['family']==family][:4]]
    data.update(parent_train_ids=[r['id'] for r in previous['train']],parent_all_groups=sorted(all_groups),parent_reserved_groups=sorted({r['group'] for s in ('dev','test') for r in previous[s]}))
    stats={s:dict(count=len(data[s]),families=dict(Counter(r['family'] for r in data[s])),answer_targets=sum(r['answer_targets'] for r in data[s]),hash=fingerprint(data[s])) for s in ('train','dev','test')}
    manifest=dict(config=fingerprint(cfg),stats=stats,parent_selection=fingerprint(pm),new_available=available,new_counts=cfg['new_counts'],rehearsal_counts=cfg['rehearsal_counts'],sources={k:pm[k] for k in ('smol_source','reading_source','skill_sources')},selected_ids={s:[r['id'] for r in data[s]] for s in ('train','dev','test')},limitations='Synthetic Smol source; automated filters and sampled review do not certify every fact. Reuses untrained parent source dev/test; 24 additional development and 24 reserved manual prompts. No benchmark targets in gradients.')
    if review:
        for role in ('new','rehearsal'):
            for family in cfg['training_counts']:
                rr=[r for r in ordered if r['family']==family and r['role']==role]
                for r in rr[::max(1,len(rr)//10)][:10]:print('[sample]',__import__('json').dumps(r,ensure_ascii=False),flush=True)
    print('[selection]',__import__('json').dumps(dict(stats=stats,new_available=available)),flush=True)
    return data,manifest

def rebuild(cfg,cancelled=lambda:False):
    global _CACHE
    data,manifest=build(cfg,cancelled)
    _CACHE=None
    if fingerprint(manifest)!=fingerprint(read_json(DIR/'selection.json')):raise ValueError('Selection differs from reviewed stream')
    return data
