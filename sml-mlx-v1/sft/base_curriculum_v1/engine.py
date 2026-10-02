"""Ordinary equal-example assistant CE, microbatch one; fixed mixed cursor."""
from sft.conversation_ab.engine import load,optimizer,learning_rate,update,encode,arrays,objective,gradients

def batch_at(data,cfg,update):
    if not 0<=update<cfg['updates']:raise ValueError('Invalid update cursor')
    rows=data['train'][update*cfg['batch']:(update+1)*cfg['batch']]
    if len(rows)!=cfg['batch']:raise ValueError('Incomplete batch')
    return rows
