"""Standalone OpenSML-150M inference on Apple Silicon using native MLX."""
import argparse,json
from pathlib import Path
import mlx.core as mx
from native_model import TransformerConfig,TransformerLM,count_parameters
from native_tokenizer import Tokenizer
from native_utils import file_sha256

def load_model(directory):
 root=Path(directory).resolve();cfg=json.loads((root/'config.json').read_text())
 if cfg['format']!='opensml-native-mlx-v1':raise ValueError('Unsupported model format')
 if file_sha256(root/'model.safetensors')!=cfg['weights_sha256']:raise ValueError('Weights checksum mismatch')
 tok=Tokenizer(root)
 if tok.vocab_size!=cfg['model']['vocab_size']:raise ValueError('Tokenizer/model vocabulary mismatch')
 model=TransformerLM(TransformerConfig(**cfg['model']));model.load_weights(str(root/'model.safetensors'),strict=True);model.eval();mx.eval(model.parameters())
 return model,tok

def generate(model,tokenizer,prompt,max_new_tokens=128,raw_completion=False):
 if max_new_tokens<1:raise ValueError('max_new_tokens must be positive')
 text=prompt if raw_completion else f'User: {prompt}\nAssistant:'
 ids=tokenizer.encode(text)
 if not ids or len(ids)+max_new_tokens>model.cfg.max_seq_len:raise ValueError('Prompt plus output budget must fit the 2048-token context')
 generated=[];caches=None;inputs=mx.array([ids]);stop='max_new_tokens'
 for _ in range(max_new_tokens):
  logits,caches=model.step(inputs,caches=caches)
  token=int(mx.argmax(logits,axis=-1).item());generated.append(token)
  if token==tokenizer.eos:stop='eos';break
  inputs=mx.array([[token]])
 answer=generated[:-1] if generated and generated[-1]==tokenizer.eos else generated
 return {'text':tokenizer.decode(answer),'token_ids':generated,'stop_reason':stop,'prompt_tokens':len(ids),'generated_tokens':len(generated)}

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model-directory',type=Path,default=Path(__file__).resolve().parent);p.add_argument('--prompt',required=True);p.add_argument('--max-new-tokens',type=int,default=128);p.add_argument('--raw-completion',action='store_true',help='Skip the User/Assistant wrapper for base-style completion prompts');a=p.parse_args()
 model,tok=load_model(a.model_directory)
 print(json.dumps(generate(model,tok,a.prompt,a.max_new_tokens,a.raw_completion),ensure_ascii=False,indent=2))
if __name__=='__main__':main()
