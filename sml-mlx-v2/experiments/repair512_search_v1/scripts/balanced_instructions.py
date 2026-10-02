"""Whole public TRAIN answers: fresh harder instructions and word-count replay."""
import json
import random
import re
import sys
from collections import Counter,defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
PUBLIC=ROOT/'experiments/public_repair_384_v1'
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(PUBLIC/'scripts'))
import data as rules
from sml_v2.common import read_json,file_sha256,fingerprint
from sml_v2.tokenization import Tokenizer

def parse_checks(prompt,labels,max_word_count):
    checks=rules.parse_checks(prompt,labels)
    if checks is not None or max_word_count<=120:return checks
    if 'length constraints:number of words' not in labels:return None
    # Apply the original conservative parser, changing only its word-count ceiling.
    # The training prompt is never altered; the original number is restored in metadata.
    pattern=r'(?:(exactly|at least|at most|less than|fewer than|more than|under|no more than|no fewer than)\s+)?('+rules.NUM+r')\s+(?:short\s+|distinct\s+)?words?'
    matches=list(re.finditer(pattern,prompt.lower()))
    if len(matches)!=1:return None
    match=matches[0];number=rules.number(match.group(2))
    if not 120<number<=max_word_count:return None
    start,end=match.span(2)
    adapted=prompt[:start]+'120'+prompt[end:]
    checks=rules.parse_checks(adapted,labels)
    if checks is None:return None
    for check in checks:
        if check['family']=='words':check['number']=number
    return checks

def pool(cap=768,max_word_count=120):
    import pyarrow.parquet as pq
    pin=read_json(PUBLIC/'source_pins.json')['allenai/tulu-3-sft-personas-instruction-following']
    assert file_sha256(pin['local'])==pin['sha256']
    tok=Tokenizer(ROOT/'tokenizer/bytebpe32k_v1')
    denied,dgrams=rules.exclusions();held=set();seen=set()
    datasets=[read_json(p)['data'] for p in (
        PUBLIC/'data/prepared.json',ROOT/'sft/conversation_foundation_v1/prepared.json',
        ROOT/'sft/text_followup_512_v1/prepared.json',ROOT/'experiments/unified_text_v2_sft_v1/data/prepared.json')]
    for dataset in datasets:
        for split in ('dev','test'):
            for row in dataset[split]:
                seen.add(rules.group_id(row['messages']))
                held.update(g for m in row['messages'] if m['role']=='user' for g in rules.grams(m['content']))
    seen.update(rules.group_id(r['messages']) for r in datasets[0]['train'])
    seen.update(rules.group_id(r['messages']) for r in datasets[-1]['train'][:384*16])
    counts=Counter();rejected=Counter();eligible=[]
    for r in pq.read_table(pin['local']).to_pylist():
        checks=parse_checks(r['prompt'],r['constraints'],max_word_count)
        if not checks:rejected['unsupported_or_ambiguous']+=1;continue
        messages=r['messages'];group=rules.group_id(messages)
        if group in seen:rejected['existing_training_or_holdout']+=1;continue
        if rules.basic_quality(messages) or rules.SCOPE.search(' '.join(m['content'] for m in messages)):
            rejected['scope_or_structure']+=1;continue
        texts=[m['content'] for m in messages if m['role']=='assistant']
        if any(rules.BOILERPLATE.search(t) for t in texts):rejected['boilerplate']+=1;continue
        if any(max(Counter(tuple(w[i:i+8]) for i in range(len(w)-7)).values(),default=0)>=2
               for w in [rules.norm(t).split() for t in texts]):rejected['repetitive_target']+=1;continue
        ag=set().union(*(rules.grams(m['content']) for m in messages))
        ug=set().union(*(rules.grams(m['content']) for m in messages if m['role']=='user'))
        if any(rules.norm(m['content']) in denied for m in messages) or ag&dgrams or ug&held:
            rejected['benchmark_or_holdout_overlap']+=1;continue
        row=dict(source='instructions',messages=messages,checks=checks,group=group,
                 id=fingerprint(messages),fresh_balanced=True,
                 origin=dict(dataset='allenai/tulu-3-sft-personas-instruction-following',id=r['id'],constraints=r['constraints']))
        try:encoded=rules.turns(tok,row,2048)
        except OverflowError:rejected['context_overflow']+=1;continue
        if max(e['targets'] for e in encoded)>cap:rejected['long_complete_answer']+=1;continue
        if not all(rules.verify(messages[-1]['content'],checks)):
            rejected['constraint_target_failed']+=1;continue
        row.update(assistant_targets=sum(e['targets'] for e in encoded),assistant_turns=len(encoded),
                   max_context=max(len(e['x']) for e in encoded))
        eligible.append(row);seen.add(group);counts.update({c['family'] for c in checks})
    fresh_count=len(eligible)
    # Include short word-count replay alongside any fresh longer count examples.
    word_replay=[r for r in datasets[0]['train'][:2048]
                 if r['source']=='instructions' and any(c['family']=='words' for c in r['checks'])]
    for row in word_replay:
        assert all(rules.verify(row['messages'][-1]['content'],row['checks']))
        eligible.append(dict(row,fresh_balanced=True,instruction_replay=True))
        counts.update({c['family'] for c in row['checks']})
    receipt=dict(source_pin=pin,maximum_complete_answer_tokens=cap,maximum_named_word_count=max_word_count,eligible=len(eligible),
                 fresh_eligible=fresh_count,word_count_parent_replay=len(word_replay),
                 families=dict(counts),rejections=dict(rejected),targets_truncated=False,
                 public_benchmark_answers_used_for_training=False,
                 limitation='Named constraints mechanically verified. Public answer content is preserved, not independently fact checked.')
    return eligible,receipt

def select(rows,exposures,seed):
    """Cycle family buckets without replacement until exhausted; then reuse explicitly."""
    rng=random.Random(seed);buckets=defaultdict(list)
    for r in rows:
        for family in sorted({c['family'] for c in r['checks']}):buckets[family].append(r)
    for family in sorted(buckets):rng.shuffle(buckets[family])
    families=sorted(buckets);offsets=Counter();selected=[];assigned=Counter();seen=set()
    assert len(families)==10
    # Rare families retain equal exposure, rather than disappearing late in a pass.
    for i in range(exposures):
        f=families[i%len(families)];v=buckets[f]
        in_batch={r['group'] for r in selected[i-i%4:]}
        row=v[offsets[f]%len(v)];offsets[f]+=1
        for _ in range(len(v)):
            if row['group'] not in in_batch:break
            row=v[offsets[f]%len(v)];offsets[f]+=1
        assert row['group'] not in in_batch
        selected.append(dict(row,instruction_stratum=f));assigned[f]+=1;seen.add(row['group'])
    return selected,dict(family_exposures=dict(assigned),unique_instruction_conversations=len(seen),
                         instruction_exposures=exposures,explicit_reuse=True,
                         covered_named_families=dict(Counter(c['family'] for r in selected for c in r['checks'])))

if __name__=='__main__':
    rows,receipt=pool();_,selection=select(rows,1024,202610011)
    print(json.dumps(dict(pool=receipt,selection=selection),indent=2))
