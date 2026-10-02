"""Side-by-side full benchmark reports against preserved 768; no automatic selection."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from sml_v1.common import read_json,atomic_json
EXP=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--step',type=int,required=True);a=p.parse_args()
    ref=ROOT/'diagnostics/text_followup_512_benchmarks_v1/text-followup-512-768'
    new=EXP/f'benchmarks/unified-text-v1-{a.step}'
    lines=['# Unified SFT vs retained 768','','Higher benchmark accuracy is better. Changes are percentage points.','','| Metric ↑ | Retained 768 | New checkpoint | Change |','|---|---:|---:|---:|'];values=[]
    for suite in ('multiple-choice','ifeval'):
        base=read_json(ref/suite/'summary.json');candidate=read_json(new/suite/'summary.json')
        if any(r['status']!='complete' or r['completed']!=r['expected'] for r in (base,candidate)):raise ValueError('Full completed suites required')
        if suite=='multiple-choice':
            pairs=[(f'{task} {key}',base['tasks'][task][key],candidate['tasks'][task][key]) for task in base['tasks'] for key in ('acc','acc_norm')]
        else:pairs=[(f'IFEval {mode} {level}',base[mode][level+'_accuracy'],candidate[mode][level+'_accuracy']) for mode in ('strict','loose') for level in ('prompt','instruction')]
        for name,x,y in pairs:
            values.append(dict(metric=name,baseline=x,new=y,change_pp=100*(y-x)))
            lines.append(f'| {name} | {100*x:.2f}% | {100*y:.2f}% | {100*(y-x):+.2f} pp |')
    lines+=['','Benchmark improvements must also be checked against matched conversational retention and saved answers. No overall winner is automatically selected.']
    path=EXP/f'comparisons/step_{a.step:05d}.md';path.parent.mkdir(parents=True,exist_ok=True);path.write_text('\n'.join(lines)+'\n')
    atomic_json(path.with_suffix('.json'),dict(step=a.step,baseline='retained-768',metrics=values,automatic_selection=False))
    print('\n'.join(lines));print('Saved:',path)


if __name__=='__main__':main()
