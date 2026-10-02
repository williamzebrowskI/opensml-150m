"""Exploratory, predeclared development and public benchmark gates."""
import json
import sys
from pathlib import Path
EXP=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text())

def compare(name,updates=None):
    c=EXP/'candidates'/name;cfg=read(c/'config.json')
    target=updates or cfg['updates']
    base=read(EXP/'baseline.json');ev=read(c/f"runs/sft/evaluations/update_{target:05d}.json")
    br=base['development']['retention'];bp=base['development']['public_development']
    cr=ev['retention']['metrics'];cp=ev['public_development']['metrics']
    gate={
        'public_repetition':cp['repeated']<=bp['repeated'],
        'legacy_repetition':cr['repeated']<=br['repeated']+1/br['generated_turns']+1e-9,
        'followup_joint':cr['text_followup_joint_proxy']>=br['text_followup_joint_proxy']-1/32-1e-9,
        'public_constraints':cp['verified_named_constraint_passes']>=bp['verified_named_constraint_passes'],
        'public_stop':cp['stopped']>=bp['stopped']-2/bp['generated_turns']-1e-9,
        'chat_repetition':cr['chat_repeated']<=br['chat_repeated'],
    }
    result=dict(candidate=name,updates=target,development_gate=gate,
                development_pass=all(gate.values()),changes={})
    for group,a,b in [('retention',br,cr),('public_development',bp,cp)]:
        result['changes'][group]={k:dict(baseline=a[k],candidate=b[k],difference=b[k]-a[k])
            for k in a if isinstance(a[k],(float,int)) and not isinstance(a[k],bool)}
    benchmark_dir=c/'benchmarks'/('repair512-search-'+name+(f'-u{updates}' if updates else ''))
    if all((benchmark_dir/s/'summary.json').exists() and read(benchmark_dir/s/'summary.json').get('status')=='complete' for s in ('multiple-choice','ifeval')):
        mc=read(benchmark_dir/'multiple-choice/summary.json');ife=read(benchmark_dir/'ifeval/summary.json')
        bm=base['benchmarks']['multiple-choice'];bi=base['benchmarks']['ifeval']
        assert mc['completed']==15428 and ife['completed']==541 and mc['status']==ife['status']=='complete'
        diffs={task:{metric:mc['tasks'][task][metric]-bm['tasks'][task][metric] for metric in ('acc','acc_norm')} for task in bm['tasks']}
        result['benchmark_changes']=dict(multiple_choice=diffs,ifeval={kind:{metric:ife[kind][metric]-bi[kind][metric]
            for metric in ('prompt_accuracy','instruction_accuracy')} for kind in ('strict','loose')})
        result['benchmark_gate']=dict(strict_prompt=ife['strict']['prompt_accuracy']>=bi['strict']['prompt_accuracy'],
            strict_instruction=ife['strict']['instruction_accuracy']>=bi['strict']['instruction_accuracy'],
            loose_prompt=ife['loose']['prompt_accuracy']>=bi['loose']['prompt_accuracy'],
            loose_instruction=ife['loose']['instruction_accuracy']>=bi['loose']['instruction_accuracy'],
            mc_mean=sum(d['acc_norm'] for d in diffs.values())>0,
            mc_no_material_regression=all(d['acc_norm']>=-.005 for d in diffs.values()))
        result['retain']=all(result['benchmark_gate'].values()) and result['development_pass']
    (c/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    return result

if __name__=='__main__':print(json.dumps(compare(sys.argv[1],int(sys.argv[2]) if len(sys.argv)>2 else None),indent=2))
