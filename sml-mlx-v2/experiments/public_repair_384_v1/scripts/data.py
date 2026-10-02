"""Existing public answers only; conservative constraint parsing and filtering."""
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from sml_v2.common import atomic_json, file_sha256, fingerprint, read_json
from sml_v2.tokenization import Tokenizer
from sft.conversation_foundation_v1.data import basic_quality, exclusions, exclusion_files, grams, group_id, norm, turns, visible_prefix

EXP = Path(__file__).resolve().parents[1]
FOUNDATION = ROOT / 'sft/conversation_foundation_v1/prepared.json'
UNIFIED = ROOT / 'experiments/unified_text_v2_sft_v1/data/prepared.json'
SCOPE = re.compile(r'```|[{}]|\b(json|xml|yaml|sql|python|javascript|programming|algebra|calculus|arithmetic|calculate|quadratic|coding|html|css|regex)\b|https?://|\d\s*[+*/=]\s*\d', re.I)
BOILERPLATE = re.compile(r"\b(?:i made (?:a few|some|the following) changes|changes (?:i made|made)|changed\s+[\"“]|i (?:cannot|can't|am unable to) (?:help|provide|assist)|please try again)\b",re.I)
LABELS = {
 'punctuation:use no comma':'no_comma', 'format:title':'title', 'use quotation':'quotation',
 'case:in english and lowercase':'lowercase', 'in english and capital':'uppercase',
 'length constraints:number of sentences':'sentences', 'length constraints:number of words':'words',
 'format:number of bullet lists':'bullets', 'length constraints:number of paragraphs':'paragraphs',
 'specific ending':'ending',
}
NUMBERS = {w:i for i,w in enumerate('zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty'.split())}
NUM = r'(?:\d+|'+'|'.join(NUMBERS)+r')'

def number(text):
    return int(text) if text.isdigit() else NUMBERS[text.lower()]

def parse_checks(prompt, labels):
    """Only retain fully supported metadata and unambiguous global count requests.

    Parsed checks validate the named constraints, not all possible semantic or
    implicit requirements. Ambiguous/per-paragraph instructions are rejected.
    """
    if not 1 <= len(labels) <= 3 or any(c not in LABELS for c in labels):
        return None
    p = prompt.lower(); checks=[]
    if re.search(r'\b(?:each|every|per) (?:sentence|paragraph|bullet|section)',p):
        return None
    for label in labels:
        k=LABELS[label];c={'family':k}
        if k=='no_comma':
            if not re.search(r'no commas?|without (?:using )?commas?|avoid (?:using )?commas?',p):return None
        elif k=='title':
            if '<<' not in prompt and not re.search(r'double (?:angular|angle) brackets',p):return None
        elif k=='quotation':
            if not re.search(r'(?:entire|whole) (?:response|answer|output|text).{0,55}(?:double )?(?:quotes|quotation marks)|(?:wrap|enclose|surround).{0,55}(?:double )?(?:quotes|quotation marks)',p):return None
        elif k=='lowercase':
            if not re.search(r'(?:all |entirely |in |and )lowercase|lowercase (?:letters|only)',p):return None
        elif k=='uppercase':
            if not re.search(r'(?:all |only |entirely |in )(?:uppercase|capital letters)|(?:entire|whole).{0,30}capital',p):return None
        elif k=='ending':
            matches=re.findall(r'(?:end|finish|conclude)[^\n]{0,65}?with\s+(?:the\s+)?(?:exact\s+)?(?:(?:phrase|sentence|words)\s*:?\s*)?["“\']([^"”\'\n]+)["”\']',prompt,re.I)
            if len(matches)!=1:return None
            c['phrase']=matches[0]
        else:
            unit={'sentences':'sentences?', 'words':'words?', 'bullets':'bullet (?:points?|lists?)', 'paragraphs':'paragraphs?'}[k]
            pattern=r'(?:(exactly|at least|at most|less than|fewer than|more than|under|no more than|no fewer than)\s+)?('+NUM+r')\s+(?:short\s+|distinct\s+)?'+unit
            matches=list(re.finditer(pattern,p))
            if len(matches)!=1:return None
            m=matches[0]; n=number(m.group(2)); relation=m.group(1) or 'exactly'
            if not 1<=n<=(120 if k=='words' else 8):return None
            # A bare number range is not an exact count.
            if re.search(r'\d\s*[-–]\s*$',p[max(0,m.start()-8):m.start()]):return None
            c.update(number=n,relation=relation)
            if k=='paragraphs':c['separator']='divider' if '***' in prompt else 'blank_line'
        checks.append(c)
    return checks

def verify(text, checks):
    from evaluation.full_benchmarks.core import runtime
    runtime()
    from lm_eval.tasks.ifeval import instructions_util
    t=text.strip();result=[]
    for c in checks:
        k=c['family'];ok=False
        if k=='no_comma':ok=',' not in t
        elif k=='title':ok=bool(re.search(r'<<[^\n<>]*\S[^\n<>]*>>',t))
        elif k=='quotation':ok=len(t)>1 and t[0]==t[-1]=='"'
        elif k=='lowercase':ok=t==t.lower() and any(x.isalpha() for x in t)
        elif k=='uppercase':ok=t==t.upper() and any(x.isalpha() for x in t)
        elif k=='ending':
            ending=t[1:-1] if any(v['family']=='quotation' for v in checks) and len(t)>1 and t[0]==t[-1]=='"' else t
            ok=ending.endswith(c['phrase'])
        else:
            if k=='words':n=instructions_util.count_words(t)
            elif k=='sentences':n=instructions_util.count_sentences(t)
            elif k=='bullets':n=len(re.findall(r'^\s*(?:\*[^*]|-).*$',t,re.M))
            elif k=='paragraphs':n=len([x for x in re.split(r'\s?\*\*\*\s?' if c['separator']=='divider' else r'\n\s*\n',t) if x.strip()])
            v=c['number'];r=c['relation']
            ok={'exactly':n==v,'at least':n>=v,'no fewer than':n>=v,'at most':n<=v,'no more than':n<=v,'less than':n<v,'fewer than':n<v,'under':n<v,'more than':n>v}[r]
        result.append(bool(t and ok))
    return result

def prepare(cfg):
    import pyarrow.parquet as pq
    tok=Tokenizer(ROOT/'tokenizer/bytebpe32k_v1');pins=read_json(EXP/'source_pins.json')
    for repo,p in pins.items():
        if repo=='HuggingFaceH4/no_robots':continue
        if file_sha256(p['local'])!=p['sha256']:raise ValueError('Public source bytes changed')
    denied,denied_grams=exclusions();held_grams=set();seen=set();reject=Counter();pools=defaultdict(list)
    foundation=read_json(FOUNDATION)['data'];unified=read_json(UNIFIED)['data']
    consumed=unified['train'][:384*16]
    seen.update(group_id(r['messages']) for r in consumed)
    for dataset in (foundation,unified):
        for split in ('dev','test'):
            for r in dataset[split]:
                held_grams.update(g for m in r['messages'] if m['role']=='user' for g in grams(m['content']))
                seen.add(group_id(r['messages']))
    candidates=[]
    for i,r in enumerate(foundation['train']):
        if r['source'] not in ('everyday-conversations','smol-magpie-ultra-short','ultrachat'):continue
        candidates.append(dict(source='chat',messages=r['messages'],origin=dict(dataset=r['repo'],file=r['file'],row_index=r['row_index'],parent_id=r['id'])))
    for i,line in enumerate(Path(pins['databricks/databricks-dolly-15k']['local']).read_text().splitlines()):
        r=json.loads(line);context=r['context'].strip();cat=r['category']
        source='grounded' if context and cat in ('closed_qa','information_extraction','summarization') else 'human'
        if source=='human' and cat not in ('open_qa','general_qa','classification','brainstorming','creative_writing'):continue
        prompt=r['instruction'].strip()+ ('\n\nReference passage:\n'+context if context else '')
        candidates.append(dict(source=source,messages=[dict(role='user',content=prompt),dict(role='assistant',content=r['response'].strip())],category=cat,origin=dict(dataset='databricks/databricks-dolly-15k',row_index=i)))
    for r in pq.read_table(pins['allenai/tulu-3-sft-personas-instruction-following']['local']).to_pylist():
        checks=parse_checks(r['prompt'],r['constraints'])
        if not checks:reject['ambiguous_or_unsupported_constraint']+=1;continue
        candidates.append(dict(source='instructions',messages=r['messages'],checks=checks,origin=dict(dataset='allenai/tulu-3-sft-personas-instruction-following',id=r['id'],constraints=r['constraints'])))
    random.Random(cfg['seed']).shuffle(candidates)
    for r in candidates:
        messages=r['messages'];s=r['source'];group=group_id(messages)
        if group in seen:reject['duplicate_or_parent_holdout']+=1;continue
        if basic_quality(messages) or SCOPE.search(' '.join(m['content'] for m in messages)):reject['scope_or_structure']+=1;continue
        texts=[m['content'] for m in messages if m['role']=='assistant']
        if any(BOILERPLATE.search(t) for t in texts):reject['boilerplate']+=1;continue
        # This pilot excludes repeated long spans, including deliberate repeat tasks.
        repetitive=False
        for t in texts:
            w=norm(t).split();c=Counter(tuple(w[i:i+8]) for i in range(len(w)-7))
            if max(c.values(),default=0)>=2:repetitive=True
        if repetitive:reject['repeated_eightword_span']+=1;continue
        ag=set().union(*(grams(m['content']) for m in messages))
        ug=set().union(*(grams(m['content']) for m in messages if m['role']=='user'))
        if any(norm(m['content']) in denied for m in messages) or ag&denied_grams or ug&held_grams:reject['evaluation_overlap']+=1;continue
        try:encoded=turns(tok,r,cfg['context'])
        except OverflowError:reject['context_overflow']+=1;continue
        if max(e['targets'] for e in encoded)>cfg['source_caps'][s]:reject['long_complete_answer']+=1;continue
        if s=='instructions' and not all(verify(messages[-1]['content'],r['checks'])):reject['constraint_target_failed']+=1;continue
        r.update(id=fingerprint(messages),group=group,assistant_turns=len(encoded),assistant_targets=sum(e['targets'] for e in encoded),max_context=max(len(e['x']) for e in encoded))
        pools[s].append(r);seen.add(group)
    print('[eligible]',json.dumps({s:len(p) for s,p in pools.items()}),flush=True)
    # Round-robin strata limit dominance by a single easy constraint or Dolly category.
    def stratify(rows):
        buckets=defaultdict(list)
        for r in rows:
            key=','.join(sorted(c['family'] for c in r['checks'])) if r['source']=='instructions' else r.get('category',r['origin'].get('dataset'))
            buckets[key].append(r)
        out=[]
        while any(buckets.values()):
            for key in sorted(buckets):
                if buckets[key]:out.append(buckets[key].pop())
        return out
    result={'train':[],'dev':[],'test':[]};selected={};holdout_grams=set()
    for s in cfg['training_counts']:
        pool=stratify(pools[s]);need=cfg['training_counts'][s]+2*cfg['holdout_per_source']
        if len(pool)<need:raise ValueError(f'Insufficient {s}: {len(pool)}/{need}; adjust preparation, never silently reuse rows')
        n=cfg['holdout_per_source'];result['dev']+=pool[:n];result['test']+=pool[n:2*n]
        for r in pool[:2*n]:holdout_grams.update(g for m in r['messages'] if m['role']=='user' for g in grams(m['content']))
        selected[s]=pool[2*n:]
    for s in selected:
        selected[s]=[r for r in selected[s] if not any(grams(m['content'])&holdout_grams for m in r['messages'] if m['role']=='user')][:cfg['training_counts'][s]]
        if len(selected[s])!=cfg['training_counts'][s]:raise ValueError('Insufficient after held-out phrase exclusion: '+s)
    rng=random.Random(cfg['seed']+1)
    for i in range(cfg['updates']):
        batch=[]
        for s,n in cfg['batch_counts'].items():batch+=selected[s][i*n:(i+1)*n]
        rng.shuffle(batch);result['train']+=batch
    stats={s:dict(conversations=len(rows),sources=dict(Counter(r['source'] for r in rows)),assistant_turns=sum(r['assistant_turns'] for r in rows),hash=fingerprint(rows)) for s,rows in result.items()}
    files=[Path(__file__),FOUNDATION,UNIFIED,EXP/'source_pins.json']+exclusion_files()+[Path(p['local']) for repo,p in pins.items() if repo!='HuggingFaceH4/no_robots']
    receipt=dict(config=fingerprint(cfg),stats=stats,inputs={str(p):file_sha256(p) for p in files},rejections=dict(reject),instruction_families=dict(Counter(c['family'] for r in selected['instructions'] for c in r['checks'])),instruction_combinations=dict(Counter(str(len(r['checks'])) for r in selected['instructions'])),
        limitations='Named constraints mechanically checked; natural-language parsing is conservative but not complete semantic verification. Public answers are not exhaustively fact checked. Source wording and full answers preserved. No guarantee of lower repetition or higher public benchmark scores. All sources use public TRAIN splits; benchmark data are exclusion-only; no consumed-384 conversation is replayed.')
    atomic_json(EXP/'data/prepared.json',dict(data=result,receipt=receipt))
    print('[prepared]',json.dumps(receipt['stats']),flush=True)
    print('[instruction-families]',json.dumps(receipt['instruction_families']),flush=True)
    print('[instruction-combinations]',json.dumps(receipt['instruction_combinations']),flush=True)
    return result
