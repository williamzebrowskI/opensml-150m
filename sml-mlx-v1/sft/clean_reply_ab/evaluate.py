"""Separate memorization, paraphrase transfer, new tasks, and retained skills."""
import mlx.core as mx
from collections import Counter
from sft.answer_completion.evaluate import teacher_stats,summary,normalized
from sft.reading_repair.generation import generate

def assess(b,data,cfg,cancelled=lambda:False):
    b.model.eval();result={}
    for name,key in [('trained','train_probes'),('paraphrase','paraphrases'),('new_tasks','dev'),('retention','retention')]:
        records=[]
        for row in data[key]:
            if cancelled():raise InterruptedError('Stopped during development evaluation')
            t=teacher_stats(b,row,cfg);g=generate(b,row['prompt'],cfg['max_new_tokens'])
            words=g['text'].lower().split();grams=Counter(tuple(words[i:i+5]) for i in range(len(words)-4))
            records.append(dict(**row,teacher=t,generation=g,exact_reference=normalized(g['text'])==normalized(row['answer']) and g['stop']=='end',repeated=bool(grams and max(grams.values())>=2)))
            if len(records)%32==0:print('[evaluation-progress]',name,len(records),flush=True)
        result[name]=dict(metrics=summary(records),answers=records)
    from sft.transfer_control.engine import prose_loss
    from sft.transfer_control.launch import prose_texts
    result['prose_nll']=prose_loss(b,prose_texts())
    result['interpretation']='Exact reference match is NOT semantic correctness for open answers. Paraphrases share training task content. New tasks are development only; test remains unopened by model. Review actual correctness, relevance, completion and repetition before selecting. No automatic best.'
    return result
