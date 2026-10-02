"""Use the verified conversation SFT update, with extra plain-text diagnostics."""
from sft.conversation_foundation_v1.engine import update, load, optimizer, generate, repeated
from sft.conversation_foundation_v1.engine import assess as foundation_assess
from .data import verify_text

def assess(b,data,cfg):
    result=foundation_assess(b,data,cfg)
    checked=[]
    for row in [r for r in data['dev'] if r['source']=='constraints'][:cfg.get('constraint_eval_conversations',32)]:
        history=[];turns=[]
        for m in row['messages']:
            if m['role']!='user':continue
            history.append(m);g=generate(b,history,cfg['max_new_tokens'])
            history.append(dict(role='assistant',content=g['text']));turns.append(dict(prompt=m['content'],**g))
            if not g['text'].strip():break
        score=verify_text(turns[-1]['text'],row['checks']) if len(turns)==2 else dict(format=False,content_proxy=False,joint_proxy=False)
        checked.append(dict(id=row['id'],source='constraint-followup',family=row['checks']['family'],turns=turns,checks=score))
    for key in ('format','content_proxy','joint_proxy'):
        result['metrics']['text_followup_'+key]=sum(r['checks'][key] for r in checked)/len(checked)
    result['metrics']['text_followup_conversations']=len(checked)
    result['answers']+=checked
    chat=[t for a in result['answers'] if a['source']=='conversation' for t in a['turns']]
    result['metrics']['chat_turns']=len(chat)
    result['metrics']['chat_stopped']=sum(t['stop']=='eos' for t in chat)/len(chat)
    result['metrics']['chat_repeated']=sum(repeated(t['tokens']) for t in chat)/len(chat)
    result['note']+=' Text follow-up checks use shared templates and literal phrase matching, not semantic correctness. Chat relevance requires reading saved responses.'
    return result
