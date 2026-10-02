"""Context-grouped public TRAIN selections plus exact original-1024 TRAIN replay."""
import csv,hashlib,io,random,re,json
from pathlib import Path
from collections import Counter
from sml_v1.common import read_json,fingerprint
from sft.skill_recovery_768.data import build as parent_build,exclusions,norm,encode_stats,finite_sentence
from sft.skill_balance.data import grams,BLOCK,ROLE
ROOT=Path(__file__).resolve().parents[2];DIR=Path(__file__).resolve().parent
SOURCES={
 'winogrande':dict(repo='allenai/winogrande',revision='01e74176c63542e6b0bcb004dcdea22d94fb67b5',file='winogrande_xl/train-00000-of-00001.parquet',bytes=2058506,sha256='5d8c38ad12b9a6c88f79b6e00aaf0f40781f93d4f94816f6cff2b67625d67399',license='CC-BY',license_source='https://github.com/allenai/winogrande#license'),
 'cosmosqa':dict(repo='wilburOne/cosmosqa',revision='b6eb99cca4e2a51dd28a9a6f562534872d851639',file='data/train.csv',bytes=16660449,sha256='d8d5ca1f9f6534b6530550718591af89372d976a8fc419360fab4158dee4d0b2',license='CC-BY-4.0',url='https://raw.githubusercontent.com/wilburOne/cosmosqa/b6eb99cca4e2a51dd28a9a6f562534872d851639/data/train.csv',license_source='https://huggingface.co/datasets/allenai/cosmos_qa#licensing-information')}
REJECT={'cosmosqa:3FHTJGYT8PELAFHDYFRFAYKAREGPGT##3RJSC4XJ139NTLM3Q0Z0GKO1LGA05G##A32W24TWSWXW##Blog_193391##q1_a1##3P7RGTLO6GRJPX7UZNUXIVN8ZI8KA4': 'Sample review: unsupported, ambiguous, malformed or over-specific inference.', 'cosmosqa:3OPLMF3EU5LMZXKU9TZ3R3WESQ8NLC##3IO1LGZLKAVMZPSD6QAK97KAN6W68K##APRZ7BR8C0ZMQ##Blog_827545##q1_a1##3L1EFR8WWT3FA364M8D0EPWARL3F9V': 'Sample review: unsupported, ambiguous, malformed or over-specific inference.', 'cosmosqa:3HYV4299H0UNQKNYM5NGWBGZV7N8EK##3K5TEWLKGW903LJRGKUJYOUNDCXIV4##AS5X6GRDJKWG3##Blog_763265##q1_a2##3R6RZGK0XHQY1QZ9EXMKNQGVHNSYVH': 'Sample review: unsupported, ambiguous, malformed or over-specific inference.', 'cosmosqa:3VADEH0UHCV96VRX9H9W7EF4KD5SP7##39U1BHVTDMP60QXJQMNHT04DS89T3H##APRZ7BR8C0ZMQ##Blog_394932##q1_a2##3W0KKJIARR7QCKF8IRYJU6T4FC88K9': 'Sample review: unsupported, ambiguous, malformed or over-specific inference.', 'cosmosqa:3EHIMLB7F7XZAUE0C6VMHEDCNG48HB##3L70J4KAZHKZ3M07DB2CB40XZ1ZDAE##A1VYRD3HO2WDUN##Blog_1215886##q1_a1##3FVBZG9CLLSWMU8410U7GXV1XS70HR': 'Sample review: unsupported, ambiguous, malformed or over-specific inference.', 'cosmosqa:3MVY4USGB81WJY30GKVCQF20SPDSIR##3RRCEFRB7PR6HPLDABDFV9E6A9GB4G##A1GV0UZU0T2ORS##Blog_1573049##q1_a2##3A3KKYU7P5VBMYA0L435FXC2FPMWMB': 'Sample review: unsupported, ambiguous, malformed or over-specific inference.', 'cosmosqa:3TZ0XG8CBUI6VOWHEU7U4XFO4A789A##3TXD01ZLD5F6IZVCQ5IA81ILTME4UA##A2JY56TZCEPK3J##Blog_605802##q1_a1##3J9L0X0VDH0WFER6WVZ0AB6IFN9W90': 'Sample review: unsupported, ambiguous, malformed or over-specific inference.', 'cosmosqa:3DQYSJDTYNPSZANDEBB848L0FHUXET##3WMOAN2SREC7CBTPXXJBVUED43UVNA##A3VVR8NR3ED04C##Blog_305039##q1_a2##34O39PNDK8MAVBPHVL3QEITRBCORB1': 'Sample review: unsupported, ambiguous, malformed or over-specific inference.', 'cosmosqa:3HYV4299H0UNQKNYM5NGWBGZUY98EN##36TFCYNS458SBFD2AHDGMXQTN69HXI##ARB80JPV0HQ5O##Blog_228031##q1_a1##31S7M7DAGGOKUAPE6PLF0226IM5TLN': 'Sample review: unsupported, ambiguous, malformed or over-specific inference.', 'cosmosqa:30OITAWPBQ1ZOZXAHY10HA8N7MA9HP##39LOEL67OT3N265VNOCM248QHUA38P##AUTYWXILTACCR##Blog_1401281##q1_a1##3NCN4N1H1IVPGDXP2EM95YOHA69BNP': 'Sample review: unsupported, ambiguous, malformed or over-specific inference.', 'cosmosqa:31JUPBOORPIGQFWXNPMK14URXFJ8L0##36DSNE9QZ8DG31T7HN557V3O115OJY##A2RUHO7I7Y4XFA##Blog_1148910##q1_a2##3TKSOBLOHNUNF48SZOH0E3OOU64BBQ': 'Sample review: wrong time reference, spam context or poor gold wording.', 'cosmosqa:3V0TR1NRVCGYBVIREV7HBEYB0ERA49##30ZX6P7VFBA2WU42UP780I0LXWR2JL##ALI92G1065WRG##Blog_1016980##q2_a1##3R0WOCG21ONVQ0FIDBP1EJ4BLS6DUS': 'Sample review: wrong time reference, spam context or poor gold wording.', 'cosmosqa:3C8QQOM6JRF9FL8AWBYBETO5X77ILQ##31Q0U3WYDSU2NEL4ACIXS9L78QS17V##AS5X6GRDJKWG3##Blog_393012##q1_a2##30OITAWPBSHEIOWH1WD098WV0OIH9L': 'Sample review: wrong time reference, spam context or poor gold wording.', 'winogrande:e9dd2217e7380d5910648436598eb397aa8f2fe2b6cb56d9c9c47da3c86a6b6b': 'Sample review: ambiguous relationship does not justify the selected option.', 'winogrande:eef8966c8aa81bcb846152cb50de633de784175a31186ac73937debb3991286d': 'Sample review: ambiguous relationship does not justify the selected option.', 'cosmosqa:3BFF0DJK8ZQMSCJBAWUG4M493FWSTT##3PMBY0YE2AIQO0JV5651N1V4FI79CN##A3VVR8NR3ED04C##Blog_1183113##q1_a2##306W7JMRY0C4D6ALWCXQN1YMLLK8BS': 'Sample review: gold adds unsupported specifics, wrong premise, poor wording or out-of-scope content.', 'cosmosqa:3SZYX62S5IEYOCOTLXE76F2Z0YY75V##3VE8AYVF8PCXD7KVRGPOVBLWSQFF8F##A3VVR8NR3ED04C##Blog_60054##q1_a1##3VZYA8PITQCK61O90YTYR0SCSRN502': 'Sample review: gold adds unsupported specifics, wrong premise, poor wording or out-of-scope content.', 'cosmosqa:3538U0YQ1FSTV1HO5ZCQNCTOQRK3FO##3OS4RQUCRAD1KGJVAVXYUFNOPMKBFE##A3VVR8NR3ED04C##Blog_755786##q1_a1##37PGLWGSJT4J1QRLRDAA7VNII8EIKS': 'Sample review: gold adds unsupported specifics, wrong premise, poor wording or out-of-scope content.', 'cosmosqa:3IKDQS3DQGEGCHE4JY5CBAYI6VVIC9##3QY5DC2MXUZVB4R8UJQTR33LPIVFUL##A3HII8DX9SINOU##Blog_870061##q2_a1##322ZSN9Z5IY3QRH0KVYY6JO3P6OT4W': 'Sample review: gold adds unsupported specifics, wrong premise, poor wording or out-of-scope content.', 'cosmosqa:3QI9WAYOGSPGQE8ZEJMNTCCVGAL6SZ##379J5II41RV0M04IQUEMIYM2TWLELB##A3VVR8NR3ED04C##Blog_12900##q1_a2##3MNJFORX8DIB353Z3GCJU7FGYDM5FK': 'Sample review: gold adds unsupported specifics, wrong premise, poor wording or out-of-scope content.', 'cosmosqa:3ABAOCJ4RAI621EE9V4ZN8UGRLTQMJ##3L6L49WXW3C4S94491M2SNEE1J354V##AS5X6GRDJKWG3##Blog_1350168##q2_a1##30UZJB2POJQGNTN6JLBHTYQJV3T35H': 'Sample review: gold adds unsupported specifics, wrong premise, poor wording or out-of-scope content.', 'cosmosqa:34KYK9TV1TMMWQPUAA3V4SYNWCFBSC##3RKNTXVS3PD1YFGL4XDJ0YU0Q69A4Q##A3218IMDX9KLPJ##Blog_1161223##q1_a1##3NQUW096N8MG4KF7SHSY10P2HGY9LD': 'Sample review: gold adds unsupported specifics, wrong premise, poor wording or out-of-scope content.', 'cosmosqa:3J9UN9O9J3QDJB4N52LS0CN9HAWJ04##3ZPBJO59KQZED98LJHC4SQN1CYIDHM##AE861G0AY5RGT##Blog_14772##q1_a1##3SX4X51T82N2FOS9XFMQPC4GI0WOAT': 'Sample review: gold adds unsupported specifics, wrong premise, poor wording or out-of-scope content.', 'cosmosqa:3MZ3TAMYTLL5OU4GHXJ7879WDLVRIN##3LPW2N6LKU0OEIZC4A1KDTOZTJ7U55##A1VR49TB3Q4BV8##Blog_1029623##q1_a1##3R5OYNIC2ENPFK3JIFQM9HZYGSAPTP': 'Sample review: gold adds unsupported specifics, wrong premise, poor wording or out-of-scope content.', 'cosmosqa:30ZKOOGW2W42P7M17FCFLI30VD51A1##36W0OB37HXCHGADHBZ11JI64WGZZHA##ARB80JPV0HQ5O##Blog_1236777##q1_a1##3JUDR1D0D6P118BEFF9FG1SASFG2QU': 'Sample review: gold adds unsupported specifics, wrong premise, poor wording or out-of-scope content.', 'cosmosqa:3I6NF2WGIIAHH5ZUE0I1CSFSZU15G2##36U2A8VAG4EN8Z3R5VEFA113S8QKY4##AS5X6GRDJKWG3##Blog_87968##q2_a1##3JUDR1D0D85GVXALZDLF8ZGILJUQ2G': 'Sample review: gold adds unsupported specifics, wrong premise, poor wording or out-of-scope content.', 'cosmosqa:37Y5RYYI0P3EF4TOV8JD1R17NVCSXW##3JZQSN0I3R8XBXR6QHPCVQEL8O2GF7##A3VVR8NR3ED04C##Blog_128694##q1_a2##3P4C70TRMTVVJU2F6NY8THBPLKZGLZ': 'Sample review: gold adds unsupported specifics, wrong premise, poor wording or out-of-scope content.', 'cosmosqa:3UUIU9GZC536SLMTE258JD69VWJ5TO##317HQ483I8QNVGAIET8SO70AMY2NI8##ARB80JPV0HQ5O##Blog_1209536##q1_a1##3S8APUMBJXHYYWL1KSRW5VR0FMCBF9': 'Sample review: gold adds unsupported specifics, wrong premise, poor wording or out-of-scope content.', 'cosmosqa:3EGKVCRQFWQ4YN2WPIJKUGYGLG4BYA##3LQ8PUHQFMQVLKO7BVDFJN3E3OTIH1##AX8WLDRDXA7P9##Blog_946383##q1_a1##309D674SH1Z6P90YDODVPNU5B1TCB4': 'Sample review: unsupported inference or fragment; no relabeling.', 'cosmosqa:3TTPFEFXCTIC6GVCJRYOX0HLHSPH6N##3C5W7UE9CGOE3TED1ETTE2YPAZEXMX##AE861G0AY5RGT##Blog_762678##q1_a1##3A9LA2FRWUS4JKEF4CJEPBID72NHX4': 'Sample review: unsupported inference or fragment; no relabeling.', 'cosmosqa:3B286OTISEF9KO25X4ILPIZMQ3JAJB##3DY4FPOOA2MDWKMA67OSCGVE0B5VRG##AE861G0AY5RGT##Blog_328589##q1_a1##3UUSLRKAUNHTSP3M03WQ4RQRDHG7DR': 'Sample review: unsupported inference or fragment; no relabeling.', 'cosmosqa:3SR6AEG6W5REP05IPATGGR1EFT3HYP##3R08VXYT7DTGTQ20N3VIT1NS04R7W3##AO33H4GL9KZX9##Blog_14838##q1_a1##3OID399FXG5RDTJLYOI20LF0I3IFD6': 'Sample review: unsupported inference or fragment; no relabeling.', 'cosmosqa:3KG2UQJ0ML2CKJZELWJZ0NJGW9QNQZ##36WLNQG782PO9RI6ZHPS4VQXOXIEBW##A3VVR8NR3ED04C##Blog_260412##q1_a2##3LN3BXKGC29BUOEEA15A2SS5J5FWGC': 'Sample review: unsupported inference or fragment; no relabeling.', 'cosmosqa:3SV8KD29L66KQ5F2BFAWZOEG7LIKZ5##3DHE4R9OCZQS5SE1MJM535857RA2GT##A35BKXOE6C5ITG##Blog_970777##q2_a1##3E24UO25Q141SMG6725E972UMQK6OH': 'Sample review: unsupported inference or fragment; no relabeling.', 'cosmosqa:3D0LPO3EABXOHHR5TNO600YFSC3OY1##3PEIJLRY6URA858R5GZGN00G6ATWX4##A37IJGXG1VEBEK##Blog_683790##q1_a1##3OKP4QVBP4BENBBB9W9IK3VER81GAI': 'Sample review: unsupported driving detail or drug-related premise.', 'cosmosqa:39WICJI5ATQZ0I8O6B6XQ4962WE3ZC##3TUI152ZZCLXY7RCMEF302ETA9Z1Q6##A3VVR8NR3ED04C##Blog_1514085##q1_a2##3RSBJ6YZECOIHQAAOXWIBFRZMLVFOH': 'Sample review: unsupported driving detail or drug-related premise.'}
SCOPE=re.compile(r'\b(sex|sexual|pregnant|pregnancy|tumor|cancer|miscarriage|abortion|abort|ultrasound|medication|meds|astrology|astrological|numerology|suicide|murder|murderer|killer|shot|rape|porn|naked|genitals|gun|guns|racist|racism|kill|killed|killing|disease|diagnosis|prescription|vaccine|religion|bible|jesus|god|allah)\b',re.I)
NONANSWER=re.compile(r'none of|all of the|cannot be determined|can not be determined|not enough (information|details)',re.I)

def clean(s):
 s=' '.join(s.split());s=re.sub(r'\s+([,.!?;:])',r'\1',s)
 s=re.sub(r"\s+'(s|t|re|ve|ll|d|m)\b",r"'\1",s)
 s=re.sub(r"\b(\w+) n['’]?t\b",r"\1n't",s)
 s=re.sub(r"\b([Ii])\s+'\s*m\b",r"\1'm",s)
 return s

def complete_sentence(text):
 # Include past-tense predicates; the older CommonGen screen is present-tense only.
 from sft.skill_balance.data import language_tools
 _,tagger=language_tools()
 tagged=tagger.tag(re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|[^\w\s]",text))
 return any(tag in ('VBP','VBZ','VBD','MD') and any(t.startswith('NN') or t=='PRP' for _,t in tagged[:i]) for i,(_,tag) in enumerate(tagged))

def stream(source,cancelled=lambda:False):
 import requests,pyarrow.parquet as pq
 spec=SOURCES[source];url=spec.get('url') or f"https://huggingface.co/datasets/{spec['repo']}/resolve/{spec['revision']}/{spec['file']}"
 buf=io.BytesIO();sha=hashlib.sha256();size=0
 with requests.get(url,stream=True,timeout=(20,90)) as res:
  res.raise_for_status()
  for chunk in res.iter_content(65536):
   if cancelled():raise InterruptedError()
   size+=len(chunk)
   if size>spec['bytes']:raise ValueError('Source too large')
   sha.update(chunk);buf.write(chunk)
 if size!=spec['bytes'] or sha.hexdigest()!=spec['sha256']:raise ValueError('Source changed: '+source)
 buf.seek(0)
 if source=='winogrande':
  for batch in pq.ParquetFile(buf).iter_batches(batch_size=128):
   if cancelled():raise InterruptedError()
   yield from batch.to_pylist()
 else:
  text=io.TextIOWrapper(buf,encoding='utf-8')
  for row in csv.DictReader(text):
   if cancelled():raise InterruptedError()
   yield row
 buf.close();print('[source-verified]',source,flush=True)

def convert(source,raw,denied,dgrams):
 if source=='winogrande':
  sentence=clean(raw['sentence']);options=[clean(raw['option1']),clean(raw['option2'])]
  if raw['answer'] not in ('1','2') or sentence.count('_')!=1:return None,'schema'
  gold=int(raw['answer'])-1;before,after=sentence.split('_')
  if not 5<=len(sentence.split())<=40 or len(before.split())<3 or len(after.split())<2:return None,'length'
  if any(not 1<=len(o.split())<=4 for o in options):return None,'length'
  # Counterfactual pairs usually differ after the blank. Group by the whole
  # normalized prefix, replacing names/options so these pairs stay together.
  group=norm(before)
  for option in sorted(options,key=len,reverse=True):group=re.sub(r'\b'+re.escape(norm(option))+r'\b','entity',group)
  group='winogrande:'+fingerprint(group)
  sid='winogrande:'+fingerprint(sentence)
  prompt='Complete the blank with the correct option. Return only that option.\nSentence: '+sentence+'\nOptions: '+'; '.join(options)
  row=dict(id=sid,group=group,source=source,question=sentence,prompt=prompt,answer=options[gold],ranking_prompt=before.rstrip(),choices=[o+after for o in options],gold=gold)
  texts=[sentence]+options+[sentence.replace('_',o) for o in options]
 else:
  context=clean(raw['context']);q=clean(raw['question']);choices=[clean(raw['answer'+str(i)]) for i in range(4)]
  if str(raw['label']) not in ('0','1','2','3'):return None,'schema'
  gold=int(raw['label']);answer=choices[gold]
  if NONANSWER.search(answer):return None,'nonanswer_gold'
  if not 12<=len(context.split())<=110 or not 4<=len(q.split())<=25 or not 5<=len(answer.split())<=32:return None,'length'
  if not complete_sentence(answer):return None,'answer_fragment'
  if not answer.endswith(('.','!','?')):answer+='.';choices[gold]=answer
  if len(set(map(norm,choices)))!=4:return None,'duplicate_choices'
  sid='cosmosqa:'+raw['id'];group='cosmosqa:'+fingerprint(norm(context))
  prefix='Context: '+context+'\nQuestion: '+q
  prompt=prefix+'\nGive the most plausible short answer using the context.'
  row=dict(id=sid,group=group,source=source,question=q,context=context,prompt=prompt,answer=answer,ranking_prompt=prefix+'\nAnswer:',choices=choices,gold=gold)
  texts=[context,q]+choices
 if sid in REJECT:return None,'sample_review'
 joined=' '.join(texts)
 if BLOCK.search(joined) or ROLE.search(joined) or SCOPE.search(joined):return None,'scope'
 if any(norm(t) in denied or grams(t)&dgrams for t in texts):return None,'public_overlap'
 return row,None

def partition(rows,cfg):
 out={s:[] for s in ('train','dev','test')}
 reviewed=read_json(DIR/'sample_review_record.json') if (DIR/'sample_review_record.json').exists() else {}
 reviewed_groups={r['group'] for r in rows if r['id'] in reviewed}
 for source in sorted({r['source'] for r in rows}):
  used=set();order=sorted([r for r in rows if r['source']==source],key=lambda r:fingerprint([cfg['seed'],r['id']]))
  for split,n in [('dev',cfg['dev_per_new_source']),('test',cfg['test_per_new_source']),('train',cfg['train_per_source'])]:
   selected=[];groups=set();ids=set()
   for r in order:
    if r['group'] in used or r['id'] in ids or (split=='test' and r['group'] in reviewed_groups):continue
    selected.append(dict(r,split=split));groups.add(r['group']);ids.add(r['id'])
    if len(selected)==n:break
   if len(selected)!=n:raise ValueError(f'{source} {split}: found {len(selected)}, need {n}')
   out[split]+=selected;used|=groups
 audit(out);return out

def audit(splits):
 for a,b in [('train','dev'),('train','test'),('dev','test')]:
  assert not({r['group'] for r in splits[a]}&{r['group'] for r in splits[b]})
  assert not({r['id'] for r in splits[a]}&{r['id'] for r in splits[b]})

def cycle(rows,n,seed):
 rng=random.Random(seed);out=[]
 while len(out)<n:
  rr=list(rows);rng.shuffle(rr);out+=rr
 return out[:n]

def build(cfg,cancelled=lambda:False):
 from sml_v1.tokenization import Tokenizer
 parent,manifest=parent_build(read_json(ROOT/'sft/skill_recovery_768/config.json'),cancelled)
 if manifest!=read_json(ROOT/'sft/skill_recovery_768/selection.json'):raise ValueError('Original replay selection changed')
 denied,dgrams,_=exclusions();pool=[];rejects=Counter()
 for source in SOURCES:
  for raw in stream(source,cancelled):
   row,reason=convert(source,raw,denied,dgrams)
   if row:pool.append(row)
   else:rejects[source+':'+reason]+=1
 split=partition(pool,cfg);u=cfg['updates'];assert cfg['train_per_source']==2*u
 new={s:cycle([r for r in split['train'] if r['source']==s],2*u,cfg['seed']+i) for i,s in enumerate(SOURCES)}
 old={s:cycle([r for r in parent['train']['commonsense'] if r['source']==s],u,cfg['seed']+i+5) for i,s in enumerate(('commonsenseqa','socialiqa'))}
 natural=list({r['id']:r for r in parent['train']['reading']}.values());fmt=list(parent['train']['instruction'])
 natural=cycle(natural,3*u,cfg['seed']+10);fmt=cycle(fmt,3*u,cfg['seed']+11)
 rank=[];targets=[];replay=[]
 for i in range(u):
  rank+=new['winogrande'][2*i:2*i+2]+new['cosmosqa'][2*i:2*i+2]+[old['commonsenseqa'][i],old['socialiqa'][i]]
  targets+=new['cosmosqa'][2*i:2*i+2]
  replay+=natural[3*i:3*i+3]+fmt[3*i:3*i+3]
 data={s:dict(parent[s]) for s in ('train','dev','test')};data['train']=dict(commonsense=rank,instruction=targets,reading=replay)
 for s in ('dev','test'):data[s]['knowledge']=split[s]
 data['anchors']=parent['anchors'];data['new_split']=split
 tok=Tokenizer(ROOT/'tokenizer/bytebpe32k_v1');stats={}
 for s in ('train','dev','test'):
  stats[s]={}
  for f,rr in data[s].items():
   sizes=[encode_stats(tok,r,cfg['context']) for r in rr]
   stats[s][f]=dict(exposures=len(rr),unique=len({r['id'] for r in rr}),hash=fingerprint(rr),maximum_tokens=max(n for n,_ in sizes),answer_tokens=sum(t for _,t in sizes),sources=dict(Counter(r.get('source','authored') for r in rr)))
 for f,n in cfg['per_update'].items():assert len(data['train'][f])==n*u
 return data,dict(config=fingerprint(cfg),sources=SOURCES,tokenizer=tok.fingerprint,parent_selection=fingerprint(manifest),stats=stats,rejections=dict(rejects),eligible_pool=dict(Counter(r['source'] for r in pool)),new_ids={s:[r['id'] for r in rr] for s,rr in split.items()},new_hashes={s:fingerprint(rr) for s,rr in split.items()},review_rejections=REJECT,anchor_hash=fingerprint(data['anchors']),limitations='Filtered source annotations plus sample review, not exhaustive correctness verification. Context grouping and exact/13-gram public-overlap rejection are heuristic, not complete semantic decontamination. Reserved sources are TRAIN subpartitions; previous retention probes reused. WinoGrande teaches ranking only; CosmosQA also teaches short natural answers. No public validation/test items used for new training.')
