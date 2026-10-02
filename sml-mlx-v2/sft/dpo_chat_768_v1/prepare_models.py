"""Generate a small review queue from frozen 768; never train or auto-label it."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from sml_v2.common import atomic_json,read_json,file_sha256,lock
from sft.conversation_foundation_v1.engine import load,generate
from sft.dpo_chat_768_v1.data import DIR,pair_id,eligible,encode_reply,exclusions,grams,norm

def main():
    cfg=read_json(DIR/'config.json');source=ROOT/cfg['source_bundle']/'model.safetensors'
    assert file_sha256(source)==cfg['source_model_sha256']
    denied,dgrams=exclusions()
    from sft.transfer_control.launch import preflight
    preflight()
    with lock(ROOT/'sft/.experiment.lock'):
        b=load(cfg);rows=[]
        for r in read_json(DIR/'authored.json'):
            assert eligible(r['messages']+[dict(role='assistant',content=r['chosen'])])
            assert not any(norm(m['content']) in denied or grams(m['content'])&dgrams for m in r['messages'])
            encode_reply(b.tokenizer,r['messages'],r['chosen'],cfg['context'])
            g=generate(b,r['messages'],cfg['max_new_tokens'])
            row=dict(id=pair_id(r['messages']),source='model-768',messages=r['messages'],chosen=r['chosen'],rejected=g['text'].strip(),generation=g,reference_review=r['review'],topic=r['topic'])
            rows.append(row);print('[sample]',len(rows),r['topic'],g['stop'],flush=True)
        atomic_json(DIR/'model_candidates.json',dict(source_sha256=cfg['source_model_sha256'],authored_sha256=file_sha256(DIR/'authored.json'),pairs=rows))
    assert file_sha256(source)==cfg['source_model_sha256']
if __name__=='__main__':main()
