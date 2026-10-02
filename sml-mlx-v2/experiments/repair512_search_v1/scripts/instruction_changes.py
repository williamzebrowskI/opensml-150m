"""Report paired official IFEval outcomes; never feeds training selection."""
import json
import sys
from collections import defaultdict
from pathlib import Path

EXP=Path(__file__).resolve().parents[1]
ROOT=EXP.parents[1]
BASE=ROOT/'experiments/public_repair_384_v1/benchmarks/public-repair-384-plus-128/ifeval/records.jsonl'

def rows(path):
    items=[json.loads(line) for line in path.read_text().splitlines()]
    assert len(items)==541
    result={r['id']:r for r in items};assert len(result)==541
    return result

def report(name):
    c=EXP/'candidates'/name
    before=rows(BASE);after=rows(c/'benchmarks'/('repair512-search-'+name)/'ifeval/records.jsonl')
    assert before.keys()==after.keys()
    result={}
    for kind in ('strict','loose'):
        families=defaultdict(lambda:dict(total=0,baseline=0,candidate=0,gained=0,lost=0))
        for key in before:
            a=before[key][kind];b=after[key][kind]
            assert a['instruction_id_list']==b['instruction_id_list']
            assert len(a['instruction_id_list'])==len(a['follow_instruction_list'])==len(b['follow_instruction_list'])
            for f,x,y in zip(a['instruction_id_list'],a['follow_instruction_list'],b['follow_instruction_list']):
                item=families[f];item['total']+=1;item['baseline']+=int(x);item['candidate']+=int(y)
                item['gained']+=int(y and not x);item['lost']+=int(x and not y)
        assert sum(v['total'] for v in families.values())==834
        result[kind]=dict(sorted(families.items()))
    p=c/'comparison.json';comparison=json.loads(p.read_text());comparison['paired_instruction_changes']=result
    p.write_text(json.dumps(comparison,indent=2)+'\n')
    return result

if __name__=='__main__':print(json.dumps(report(sys.argv[1]),indent=2))
