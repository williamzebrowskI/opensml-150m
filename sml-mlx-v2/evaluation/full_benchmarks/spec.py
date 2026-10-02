"""Pinned evaluation-only identities and public protocol."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
DIR=Path(__file__).resolve().parent
CACHE=DIR/'_cache'
RESULTS=ROOT/'diagnostics/full_benchmarks_v1'
HARNESS='b954108c9baaaa934b4ad842033b31a97ee30816'
TOKENIZER='1194292c6d906ac19e6f01c6ad2b42825143fad426a22aa8856d903776701ecb'
BASE='runs/stage_b_prose_v1/step_0073243_7f3070237eea'
MODELS={
 'sft-896-full-context':dict(bundle='runs/sft_reading_repair_v1/step_0000896_b6e5f5f75873',step=896,sha256='021bc6da816aa2b7b657a144e28b18065159554d666ba2b2159a87d0f4d66a69',decode_mode='full_context'),
 'sft-896':dict(bundle='runs/sft_reading_repair_v1/step_0000896_b6e5f5f75873',step=896,sha256='021bc6da816aa2b7b657a144e28b18065159554d666ba2b2159a87d0f4d66a69'),
 'sft-448':dict(bundle='runs/sft_response_repair_v1/step_0000448_69a0ef1bc1cf',step=448,sha256='dd1d3c26dfc43a1ac44fa63947273ec5cbebbdb1730c434e1fc36ac20abcde63'),
 'sft-384':dict(bundle='runs/sft_response_expansion_v1/step_0000384_745d0582d31c',step=384,sha256='880b775939f66931503b65ae2e3bb03e9a8b911d14723bdf0466fb1860016817'),
 'pretrained':dict(bundle=BASE,step=73243,sha256='ca9bd5d82005f28e77f319d3a5a29f29fb4e4f86e7ceea049f01162fd8aae80f')}
DATA={
 'arc_easy':dict(repo='allenai/ai2_arc',revision='210d026faf9955653af8916fad021475a3f00453',file='ARC-Easy/test-00000-of-00001.parquet',split='test',rows=2376,sha256='4160597d618ae851c7eb04e281574f3f654776216ac6b6641588d64527b47177'),
 'arc_challenge':dict(repo='allenai/ai2_arc',revision='210d026faf9955653af8916fad021475a3f00453',file='ARC-Challenge/test-00000-of-00001.parquet',split='test',rows=1172,sha256='62f03257e737aed263f55c6abf87c7bb0028a44a6bdd2a26eb1279eb42c1d1e9'),
 'piqa':dict(repo='baber/piqa',revision='142f6d7367fd9877f0fb3b5734ea6a545f54cdd1',file='piqa_validation.parquet',split='validation',rows=1838,sha256='2b17ee1215b1ceb2ddabf4c90a05c5a8b833bef9d207960f57d111c43a2080f3'),
 'hellaswag':dict(repo='Rowan/hellaswag',revision='218ec52e09a7e7462a5400043bb9a69a41d06b76',file='data/validation-00000-of-00001.parquet',split='validation',rows=10042,sha256='899813071e1e95efafec90f856e1987d2150fa4d020fc005df6962c259f660cd'),
 'ifeval':dict(repo='google/IFEval',revision='966cd89545d6b6acfd7638bc708b98261ca58e84',file='ifeval_input_data.jsonl',split='train',rows=541,sha256='6a85310ca8ce15eff755aa08a3a4ff931c7e273e7515ebb3c492ea85fd8288f2')}
MC=tuple(k for k in DATA if k!='ifeval')
PUNKT_REVISION='4f15a3d89eefe9748ec1c05be495d91289197155'
MAX_CONTEXT=2048
GENERATION_LIMIT=1280
SEED=24092026
