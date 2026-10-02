"""Save compact trial provenance and results before authorized failed-run cleanup."""
import json,sys
from pathlib import Path
EXP=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(Path(__file__).parent))
from experiment import delete,verify_parent
from compare import compare
from sml_v2.common import atomic_json,read_json
name=sys.argv[1];c=EXP/'candidates'/name
assert not (c/'KEEP.json').exists(), 'User-retained candidate: cleanup is forbidden without a new explicit deletion instruction.'
r=compare(name);assert not r.get('retain',False)
extra={k:read_json(c/(k+'.json')) for k in ('selection','source_pins') if (c/(k+'.json')).exists()}
extra['fresh_development']={p.stem:read_json(p)['metrics'] for p in (c/'runs/sft/evaluations').glob('targeted_*.json')}
extra['reason']='Failed predeclared development gates; full benchmarks skipped.' if not r['development_pass'] and 'benchmark_gate' not in r else 'Failed predeclared overall acceptance gates.'
delete(c)
p=EXP/'comparisons'/f'{name}.json';receipt=read_json(p);receipt.update(extra);atomic_json(p,receipt)
state=read_json(EXP/'results.json');state['trials']=[t for t in state['trials'] if t['candidate']!=name]+[receipt]
state.update(trials_completed=len(state['trials']),completed_trials=len(state['trials']),active_candidate=None,status='continuing_dataset_search',all_failed_candidate_directories_removed=True)
atomic_json(EXP/'results.json',state)
print(json.dumps(dict(candidate=name,development=r['development_gate'],benchmark=r.get('benchmark_gate'),removed=True,fresh_development=extra['fresh_development']),indent=2))
