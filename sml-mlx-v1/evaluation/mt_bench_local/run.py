"""Official MT-Bench questions, native OpenSML answers, local Prometheus helpfulness judging."""
import argparse
import ast
from collections import Counter, defaultdict
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from sml_v1.common import atomic_json, file_sha256, fingerprint, lock, read_json

RESULTS = ROOT / 'diagnostics/mt_bench_local_v1'
BASE = ROOT / 'runs/stage_b_prose_v1/step_0073243_7f3070237eea/model.safetensors.json'
TOKENIZER = '1194292c6d906ac19e6f01c6ad2b42825143fad426a22aa8856d903776701ecb'
MODELS = {
    '768': ('two-turn-high640-768', 'runs/sft_two_turn_640_v1/step_0000768_49364940d7ce', 'db6b16b45153f814de3c0799c49e6460e6b638982168ffcb7bfc2b6b1878bce1'),
    '2688': ('preference-2688', 'runs/sft_preference_long_640_v1/step_0002688_d8a8efb041d1', 'dcf32522b9f045deb3e1766dc54571735074f8265c85549562f4c03979f22d4a'),
    '640': ('grounded-rank-640', 'runs/sft_grounded_rank_384_v1/step_0000640_7b477e0e5eea', 'cfc53b48c0a0810cbe7e41e0457414096a498538bdd0962f6f46d13b5164fe69'),
    '1920': ('skill-balance-1920', 'runs/sft_skill_balance_v1/step_0001920_628a0a65099b', '4cac7f733e6ef63230e26e2aa66418bf0d9661ef55f46fa1e01f2ac602b17407'),
    '2048': ('constraint-completion-2048', 'runs/sft_constraint_completion_1920_v1/step_0002048_1267d9f4cf7a', 'e4e8a0de28e2141f6a580b49cc950aa64492c9f94d0d3aa98e6a26a2f439764b'),
}
TEMPERATURES = dict(writing=.7, roleplay=.7, extraction=0., math=0., coding=0., reasoning=0., stem=.1, humanities=.1)
PROTOCOL = dict(name='MT-Bench / local Prometheus helpfulness', version=1, questions=80, turns=160,
    answer_max_tokens=1024, answer_context=2048, temperatures=TEMPERATURES, top_k=50, top_p=1.,
    seed_per_question=0, choices=1, repetition_penalty=1., answer_template='native-visible-prefix',
    judge_max_tokens=1024, judge_temperature=0., judge_scale=[1,5], rubric='upstream HELPFULNESS_RUBRIC',
    references='upstream GPT-4 references for math/reasoning/coding; no-reference prompt elsewhere',
    official_gpt4_score=False, training=False)


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def prompt_constants():
    # Read only string constants from the pinned upstream file; never execute it.
    result = {}
    for node in ast.parse((DIR/'upstream/prompts.py.txt').read_text()).body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            value = node.value
            strip = isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute) and value.func.attr == 'strip'
            if strip: value = value.func.value
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                result[node.targets[0].id] = value.value.strip() if strip else value.value
    return result


def questions():
    result = rows(DIR/'upstream/question.jsonl')
    assert len(result) == 80 and len({r['question_id'] for r in result}) == 80
    assert Counter(r['category'] for r in result) == Counter({c:10 for c in TEMPERATURES})
    assert all(len(r['turns']) == 2 for r in result)
    return result


def check_assets():
    manifest = read_json(DIR/'assets.json')
    for name, digest in manifest.items():
        if file_sha256(DIR/name) != digest: raise ValueError('Asset changed: '+name)
    questions()
    return manifest


def code_identity():
    paths = [Path(__file__), ROOT/'sft/intact_smoltalk_base_pilot/data.py', ROOT/'evaluation/full_benchmarks/core.py']
    paths += list((ROOT/'sml_v1').glob('*.py'))
    return {str(p.relative_to(ROOT)):file_sha256(p) for p in paths}


def environment():
    return {p:importlib.metadata.version(p) for p in ('mlx','mlx-lm','numpy','tokenizers','transformers')}


def selected(key):
    name, relative, digest = MODELS[key]
    bundle = ROOT/relative
    if file_sha256(bundle/'model.safetensors') != digest: raise ValueError('Checkpoint weights changed')
    meta = read_json(bundle/'model.safetensors.json')
    if meta['step'] != int(key) or meta['tokenizer'] != TOKENIZER: raise ValueError('Checkpoint identity mismatch')
    return dict(model_id=name, bundle=str(bundle), weights_sha256=digest,
                metadata_sha256=file_sha256(bundle/'model.safetensors.json'))


class Journal:
    def __init__(self, path, ids):
        self.path, self.ids = Path(path), list(ids)
        self.records = []
        if self.path.exists():
            raw = self.path.read_bytes(); end = raw.rfind(b'\n')+1
            if end != len(raw):
                self.path.with_suffix('.incomplete-tail').write_bytes(raw[end:])
                with self.path.open('r+b') as f: f.truncate(end)
            self.records = [json.loads(line) for line in raw[:end].splitlines()]
        if [r['id'] for r in self.records] != self.ids[:len(self.records)] or len(self.records)>len(self.ids):
            raise ValueError('Saved result order/identity mismatch')

    def append(self, row):
        if row['id'] != self.ids[len(self.records)]: raise ValueError('Unexpected result')
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open('a') as f:
            f.write(json.dumps(row,ensure_ascii=False)+'\n'); f.flush(); os.fsync(f.fileno())
        self.records.append(row)


def answer(b, messages, temperature, limit=1024):
    import mlx.core as mx
    from mlx_lm.sample_utils import make_sampler
    from sft.intact_smoltalk_base_pilot.data import visible_prefix
    prompt = visible_prefix(messages); ids = b.encode(prompt)
    budget = min(limit, 2048-len(ids))
    if budget <= 0:
        return dict(text='',token_ids=[],prompt=prompt,prompt_tokens=len(ids),generated_tokens=0,stop_reason='context_limit')
    sampler = make_sampler(temp=temperature, top_k=50)
    outputs=[]; reason='length'
    logits, cache = b.model.logits(mx.array([ids],dtype=mx.int32)); mx.eval(logits,cache)
    for i in range(budget):
        scores=logits[:,-1,:] if logits.ndim==3 else logits
        logp=scores-mx.logsumexp(scores,axis=-1,keepdims=True)
        token=int(sampler(logp).item())
        if token==b.eos: reason='eos'; break
        outputs.append(token)
        if i+1<budget:
            logits,cache=b.model.step(mx.array([[token]],dtype=mx.int32),caches=cache);mx.eval(logits,cache)
    if reason=='length' and budget<limit:reason='context_limit'
    return dict(text=b.tokenizer.decode(outputs).strip(),token_ids=outputs,prompt=prompt,
                prompt_tokens=len(ids),generated_tokens=len(outputs),stop_reason=reason)


def export_answers(out, records, model_id):
    path=out/'model_answer'/f'{model_id}.jsonl';path.parent.mkdir(parents=True,exist_ok=True)
    data=[dict(question_id=r['id'],answer_id=r['answer_id'],model_id=model_id,
               choices=[dict(index=0,turns=[g['text'] for g in r['generations']])],tstamp=r['tstamp']) for r in records]
    temp=path.with_suffix('.tmp');temp.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in data));temp.replace(path)


def generate_answers(key, out, qq, contract):
    import mlx.core as mx
    from evaluation.full_benchmarks import core,spec
    journal=Journal(out/'answers.jsonl',[q['question_id'] for q in qq])
    export_answers(out,journal.records,contract['selected']['model_id'])
    if len(journal.records)==80:return journal.records
    name,relative,digest=MODELS[key]
    spec.MODELS[name]=dict(bundle=relative,sha256=digest,step=int(key))
    b=core.Backend(name)
    try:
        for q in qq[len(journal.records):]:
            mx.random.seed(0);messages=[];generations=[]
            for prompt in q['turns']:
                messages.append(dict(role='user',content=prompt))
                g=answer(b,messages,TEMPERATURES[q['category']]);generations.append(g)
                messages.append(dict(role='assistant',content=g['text']))
            journal.append(dict(id=q['question_id'],category=q['category'],generations=generations,
                answer_id=fingerprint([fingerprint(contract),q['question_id']])[:22],tstamp=time.time()))
            export_answers(out,journal.records,name)
            print(f'[answers] {len(journal.records)}/80 conversations',flush=True)
    finally:
        del b;gc.collect();mx.clear_cache()
    return journal.records


def make_judge_prompt(q, record, turn, reference=None):
    p=prompt_constants()
    instruction=q['turns'][0]
    if turn==1:
        instruction=('Conversation context:\nUser: '+q['turns'][0]+'\nAssistant: '+record['generations'][0]['text']+
            '\n\nLatest user request: '+q['turns'][1]+'\n\nEvaluate only the response to the latest request, using this history.')
    fields=dict(instruction=instruction,response=record['generations'][turn]['text'],rubric=p['HELPFULNESS_RUBRIC'])
    if reference is not None:
        text=reference['choices'][0]['turns'][turn]
        if turn==1:
            text='Reference first answer (context):\n'+reference['choices'][0]['turns'][0]+'\n\nReference follow-up answer:\n'+text
        fields['reference_answer']=text
    return p['ABS_SYSTEM_PROMPT'],p['ABSOLUTE_PROMPT' if reference is not None else 'ABSOLUTE_PROMPT_WO_REF'].format(**fields)


def parse_score(text):
    match=re.search(r'\[RESULT\]\s*([1-5])\s*$',text)
    return int(match.group(1)) if match else None


class Judge:
    def __init__(self):
        from mlx_lm import load
        self.model,self.tokenizer=load(str(DIR/'judge_model'),tokenizer_config={'trust_remote_code':False})

    def grade(self, q, record, turn, reference=None):
        import mlx.core as mx
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler
        system,prompt=make_judge_prompt(q,record,turn,reference)
        tokens=self.tokenizer.apply_chat_template([dict(role='system',content=system),dict(role='user',content=prompt)],
            tokenize=True,add_generation_prompt=True)
        if len(tokens)+PROTOCOL['judge_max_tokens']>32768: raise ValueError('Judge context overflow; no truncation')
        mx.random.seed(0);text='';last=None
        for last in stream_generate(self.model,self.tokenizer,tokens,max_tokens=PROTOCOL['judge_max_tokens'],sampler=make_sampler(temp=0)):
            text+=last.text
        return dict(score=parse_score(text),feedback=text,prompt_tokens=len(tokens),
                    generated_tokens=last.generation_tokens if last else 0)


def evaluate_judge(out, qq, answers):
    journal=Journal(out/'judgments.jsonl',[f'{q["question_id"]}:{t+1}' for q in qq for t in range(2)])
    if len(journal.records)==160:return journal.records
    references={r['question_id']:r for r in rows(DIR/'upstream/reference.jsonl')}
    judge=Judge()
    for q,record in zip(qq,answers):
        for t in range(2):
            ident=f'{q["question_id"]}:{t+1}'
            if ident in {r['id'] for r in journal.records}:continue
            ref=references.get(q['question_id']) if q['category'] in ('math','coding','reasoning') else None
            if q['category'] in ('math','coding','reasoning') and ref is None:raise ValueError('Reference missing')
            result=judge.grade(q,record,t,ref)
            if result['score'] is None:
                atomic_json(out/'invalid_judgment.json',dict(id=ident,**result))
                raise ValueError('Judge returned no valid score; saved invalid_judgment.json. No fabricated score or skipped item.')
            journal.append(dict(id=ident,question_id=q['question_id'],turn=t+1,category=q['category'],**result))
            print(f'[judge] {len(journal.records)}/160 turns; score={result["score"]}/5',flush=True)
    return journal.records


def summarize(out, qq):
    answers=Journal(out/'answers.jsonl',[q['question_id'] for q in qq]).records
    grades=Journal(out/'judgments.jsonl',[f'{q["question_id"]}:{t+1}' for q in qq for t in range(2)]).records
    groups=defaultdict(list)
    for g in grades:
        groups['all'].append(g['score']);groups[f'turn_{g["turn"]}'].append(g['score']);groups[g['category']].append(g['score'])
    complete=len(answers)==80 and len(grades)==160
    summary=dict(status='complete' if complete else 'partial',conversations=len(answers),judged_turns=len(grades),
        expected_conversations=80,expected_judged_turns=160,scale='1–5, higher is better; local Prometheus helpfulness',
        official_gpt4_score=False,overall_score=sum(groups['all'])/160 if complete else None,
        groups={k:dict(count=len(v),mean=sum(v)/len(v)) for k,v in groups.items()},
        stop_counts=dict(Counter(g['stop_reason'] for a in answers for g in a['generations'])))
    atomic_json(out/'summary.json',summary)
    lookup={g['id']:g for g in grades};lines=['# MT-Bench conversations — local Prometheus helpfulness','',
        'Scores are 1–5 and are not standard GPT-4 MT-Bench scores. Automated judgments need human review.','']
    for q,a in zip(qq,answers):
        lines += [f'## Question {q["question_id"]} — {q["category"]}','']
        for t,g in enumerate(a['generations']):
            lines += [f'### Turn {t+1}','', '**User**','',q['turns'][t],'','**Model**','',g['text'],'',f'Stop: {g["stop_reason"]}','']
            j=lookup.get(f'{q["question_id"]}:{t+1}')
            if j:lines += [f'**Judge: {j["score"]}/5**','',j['feedback'],'']
    (out/'conversations.md').write_text('\n'.join(lines))
    return summary


def check():
    assets=check_assets()
    from sml_v1.tokenization import Tokenizer
    from sft.intact_smoltalk_base_pilot.data import visible_prefix
    tok=Tokenizer(ROOT/'tokenizer/bytebpe32k_v1');assert tok.fingerprint==TOKENIZER
    qq=questions()
    lengths=[len(tok.encode(visible_prefix([dict(role='user',content=q['turns'][0])]))) for q in qq]
    assert max(lengths)<2048
    judge=Judge();q=dict(turns=['Say hello and offer to help.','What is my name?'])
    checks=[]
    for label,record,turn in [
        ('relevant',dict(generations=[dict(text='Hello! How can I help you today?')]),0),
        ('irrelevant',dict(generations=[dict(text='Facebook Ads')]),0),
        ('followup',dict(generations=[dict(text='Hello, Maya!'),dict(text='Your name is Maya.')]),1)]:
        query=q if label!='followup' else dict(turns=['My name is Maya. Say hello.','What is my name?'])
        r=judge.grade(query,record,turn);checks.append(dict(label=label,**r));print(f'[check] {label}: {r["score"]}/5',flush=True)
    assert all(r['score'] is not None for r in checks),'Judge output parsing failed'
    assert checks[0]['score']>checks[1]['score'] and checks[2]['score']>checks[1]['score'],'Judge sanity check failed'
    receipt=dict(status='passed',assets=fingerprint(assets),code=code_identity(),runtime=environment(),
        maximum_first_prompt_tokens=max(lengths),judge_checks=checks,benchmark_started=False,
        limitation='Three sanity cases verify wiring only, not judge accuracy or checkpoint quality.')
    atomic_json(DIR/'readiness.json',receipt)
    print('[ready] No checkpoint selected; no benchmark answers generated.',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    action=parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--list',action='store_true');action.add_argument('--check',action='store_true')
    action.add_argument('--run',action='store_true');action.add_argument('--summary',action='store_true')
    parser.add_argument('--model',choices=list(MODELS))
    parser.add_argument('--stage',choices=['all','answers','judge'],default='all')
    args=parser.parse_args()
    if args.list:
        for k,(name,path,_) in MODELS.items():print(f'{k:>4}  {name}: {"available" if (ROOT/path/"model.safetensors").exists() else "missing"}')
        return
    if args.summary:
        if not args.model:parser.error('--summary requires --model')
        out=RESULTS/MODELS[args.model][0]
        if not (out/'contract.json').exists():parser.error('No evaluation has been started for this checkpoint')
        with lock(DIR/'.evaluation.lock'):
            print(json.dumps(summarize(out,questions()),indent=2))
        return
    from sft.transfer_control.launch import preflight
    preflight()
    with lock(ROOT/'sft/.experiment.lock'),lock(DIR/'.evaluation.lock'):
        if args.check:check();return
        if not args.model:parser.error('--run requires an explicit --model; there is no default checkpoint')
        assets=check_assets();code=code_identity();runtime=environment();ready=read_json(DIR/'readiness.json')
        if ready['status']!='passed' or ready['assets']!=fingerprint(assets) or ready['code']!=code or ready['runtime']!=runtime:
            raise ValueError('Setup changed; rerun --check')
        contract=dict(protocol=PROTOCOL,selected=selected(args.model),assets=assets,code=code,runtime=runtime,
                      architecture_sha256=file_sha256(BASE),tokenizer=TOKENIZER)
        out=RESULTS/MODELS[args.model][0];path=out/'contract.json'
        if path.exists() and read_json(path)!=contract:raise ValueError('Existing results use a different protocol; refusing to mix')
        if not path.exists():atomic_json(path,contract)
        qq=questions()
        try:
            if args.stage in ('all','answers'):generate_answers(args.model,out,qq,contract)
            if args.stage in ('all','judge'):
                saved=Journal(out/'answers.jsonl',[q['question_id'] for q in qq]).records
                if len(saved)!=80:raise ValueError('Generate all 80 conversations before judging')
                evaluate_judge(out,qq,saved)
        finally:
            print(json.dumps(summarize(out,qq),indent=2));print(f'Results: {out}',flush=True)


if __name__=='__main__':
    try:main()
    except KeyboardInterrupt:
        print('\n[interrupted] Completed items saved. Rerun the same command to resume.');sys.exit(130)
