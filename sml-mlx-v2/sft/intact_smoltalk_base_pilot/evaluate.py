"""All 500 validation conversations; gold-history and actual-history saved answers."""
from .data import turns, review_rows, retention_rows, visible_prefix
from .engine import generate, objective, arrays
from sft.conversation_completion_2048.evaluate import repeated
from sft.reading_repair.evaluate import grade
from sft.constraint_completion_1920.constraints import verify
from sft.transfer_control.engine import prose_loss
from sft.transfer_control.launch import prose_texts
from sft.natural_control.data import norm


def assess(b,data,cfg,split='dev',cancelled=lambda:False,smoke=False):
    b.model.eval(); loss=0.; targets=0; answers=[]
    validation=data[split][:1] if smoke else data[split]
    for i,row in enumerate(validation):
        if cancelled(): raise InterruptedError('Stopped during validation')
        for e in turns(b.tokenizer,row,cfg['context']):
            x,y=arrays([e],b.pad); loss += float(objective(b.model,x,y).item())*e['targets']; targets+=e['targets']
        if (i+1)%100==0:print('[intact-validation]',i+1,'conversations',flush=True)
    selected=review_rows(data,cfg,split)[:1] if smoke else review_rows(data,cfg,split)
    for row in selected:
        actual=[]
        for i,m in enumerate(row['messages']):
            if m['role']!='assistant':actual.append(m);continue
            gold_messages=row['messages'][:i]
            gold_room=2048-len(b.encode(visible_prefix(gold_messages)))
            own_room=2048-len(b.encode(visible_prefix(actual)))
            gold_limit=min(cfg['max_new_tokens'],gold_room)
            gold=generate(b,gold_messages,gold_limit,cancelled)
            if own_room<=0:
                own=dict(text='',tokens=0,token_ids=[],stop='context_limit')
            else:
                own=gold if actual==gold_messages else generate(b,actual,min(cfg['max_new_tokens'],own_room),cancelled)
            answers.append(dict(id=row['id']+':'+str(i),source=row['source'],message_index=i,
                                source_messages=gold_messages,actual_messages=list(actual),reference=m['content'],
                                gold_history=gold,own_history=own,gold_budget=gold_limit,
                                own_budget=max(0,min(cfg['max_new_tokens'],own_room))))
            # Preserve the generated preceding reply. A context-limit failure is
            # explicit; it is not rescued by silently switching to gold history.
            actual.append(dict(role='assistant',content=own['text']))
    rr=retention_rows(); reading=[]; constraints=[]; qa=[]
    for kind,rows in rr.items():
        for row in rows[:1] if smoke else rows:
            if cancelled():raise InterruptedError('Stopped in retention checks')
            g=generate(b,[dict(role='user',content=row['prompt'])],128,cancelled)
            record=dict(**row,generation=g)
            if kind=='reading':record.update(grade(row,g['text'],g['stop']));reading.append(record)
            elif kind=='constraints':
                record['rules_pass']=g['stop']=='end' and bool(g['text'].strip()) and all(verify(g['text'],row['rules']))
                constraints.append(record)
            else:record['exact']=g['stop']=='end' and norm(g['text'])==norm(row['reference']);qa.append(record)
    generations=[r['own_history'] for r in answers]
    multi=[r['own_history'] for r in answers if sum(m['role']=='assistant' for m in r['source_messages'])>0]
    def rate(gs,key):return sum(key(g) for g in gs)/len(gs) if gs else None
    prose=prose_loss(b,prose_texts()[:2] if smoke else prose_texts())
    metrics=dict(validation_nll=loss/targets,validation_conversations=len(validation),validation_targets=targets,
                 reviewed_conversations=len(selected),generated_turns=len(answers),
                 own_history_stopped=rate(generations,lambda g:g['stop']=='end'),
                 own_history_repeated=rate(generations,lambda g:repeated(g['text'])),
                 followup_stopped=rate(multi,lambda g:g['stop']=='end'),
                 followup_repeated=rate(multi,lambda g:repeated(g['text'])),
                 mean_tokens=sum(g['tokens'] for g in generations)/len(generations),
                 reading_exact=sum(r['passed'] for r in reading)/len(reading),
                 constraint_rules=sum(r['rules_pass'] for r in constraints)/len(constraints),
                 qa_exact_diagnostic=sum(r['exact'] for r in qa)/len(qa),prose_nll=prose)
    return dict(metrics=metrics,answers=answers,reading=reading,constraints=constraints,qa=qa,
                limitation='NLL, stopping, repetition and literal matching are diagnostics, not semantic success. Review factual relations, relevance and completion of every request, including own-history follow-ups. Retention items were previously inspected; official benchmarks and reserved conversations are not used for training or selection.')


def gate(m,baseline,cfg):
    return (m['prose_nll']<=baseline['prose_nll']+cfg['candidate_prose_increase_limit']
            and m['reading_exact']>=baseline['reading_exact']-cfg['candidate_reading_drop_limit']
            and m['constraint_rules']>=baseline['constraint_rules']-cfg['candidate_format_drop_limit']
            and m['own_history_stopped']>=baseline['own_history_stopped']-cfg['candidate_stop_drop_limit'])


def review_template(result):
    from sml_v2.common import fingerprint
    return dict(evaluation=fingerprint(result),status='unreviewed',
                rubric='Score content correctness, relevance, complete request fulfillment, naturalness and coherent follow-ups. Inspect both gold-history and own-history replies; stopping alone does not pass.',
                answers=[dict(id=r['id'],correct=None,relevant=None,complete=None,natural=None,notes='')
                         for r in result['answers']])
