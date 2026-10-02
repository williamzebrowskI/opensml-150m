"""Restore pinned benchmark and local judge assets; never run an evaluation."""
import hashlib
import json
from pathlib import Path
import urllib.request
from huggingface_hub import snapshot_download

DIR=Path(__file__).resolve().parent

def main():
    s=json.loads((DIR/'sources.json').read_text())
    paths={'question.jsonl':'fastchat/llm_judge/data/mt_bench/question.jsonl',
           'reference.jsonl':'fastchat/llm_judge/data/mt_bench/reference_answer/gpt-4.jsonl',
           'gen_model_answer.py.txt':'fastchat/llm_judge/gen_model_answer.py',
           'common.py.txt':'fastchat/llm_judge/common.py','FastChat-LICENSE':'LICENSE'}
    (DIR/'upstream').mkdir(exist_ok=True)
    for name,path in paths.items():
        url=f'https://raw.githubusercontent.com/lm-sys/FastChat/{s["fastchat_revision"]}/{path}'
        if not (DIR/'upstream'/name).exists():(DIR/'upstream'/name).write_bytes(urllib.request.urlopen(url).read())
    for name,path in {'prompts.py.txt':'libs/prometheus-eval/prometheus_eval/prompts.py','Prometheus-LICENSE':'LICENSE'}.items():
        url=f'https://raw.githubusercontent.com/prometheus-eval/prometheus-eval/{s["prometheus_revision"]}/{path}'
        if not (DIR/'upstream'/name).exists():(DIR/'upstream'/name).write_bytes(urllib.request.urlopen(url).read())
    snapshot_download(s['judge_repo'],revision=s['judge_revision'],local_dir=str(DIR/'judge_model'),
                      allow_patterns=['*.json','*.safetensors','*.model','README.md'],max_workers=4)
    for relative,expected in json.loads((DIR/'assets.json').read_text()).items():
        h=hashlib.sha256()
        with (DIR/relative).open('rb') as f:
            for chunk in iter(lambda:f.read(8*1024**2),b''):h.update(chunk)
        if h.hexdigest()!=expected:raise ValueError('Asset hash mismatch: '+relative)
    print('Assets verified. Run run.py --check next; no checkpoint has been selected.')

if __name__=='__main__':main()
