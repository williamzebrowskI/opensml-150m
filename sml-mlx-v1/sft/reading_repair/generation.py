"""Verified cached greedy generation, with full-context parity checks."""
def generate(backend,prompt,limit=256):
    """Cached greedy native format, only EOS stops. No length-forcing or repetition tricks."""
    import mlx.core as mx
    ids=backend.encode('User: '+prompt+'\nAssistant:')
    if len(ids)+limit>2048:raise ValueError('Evaluation prompt cannot fit full response budget')
    was_training=backend.model.training;backend.model.eval();output=[];stop='length'
    try:
        scores,cache=backend.model.logits(mx.array([ids],dtype=mx.int32));scores=scores[:,-1,:];mx.eval(scores,cache)
        for _ in range(limit):
            if scores.ndim!=2 or scores.shape[0]!=1:raise ValueError('Expected [batch, vocabulary] generation logits')
            token=int(mx.argmax(scores[0]).item())
            if token==backend.eos:stop='end';break
            output.append(token)
            if len(output)<limit:
                scores,cache=backend.model.step(mx.array([[token]],dtype=mx.int32),caches=cache);mx.eval(scores,cache)
        return dict(text=backend.decode(output).strip(),tokens=len(output),token_ids=output,stop=stop)
    finally:backend.model.train(was_training)


def verify_generation(backend,prompts,limit=32):
    """Compare cached generation with independent full-context recomputation."""
    import mlx.core as mx
    checks=[];was_training=backend.model.training;backend.model.eval()
    try:
        for prompt in prompts:
            cached=generate(backend,prompt,limit)
            ids=backend.encode('User: '+prompt+'\nAssistant:');output=[];stop='length'
            for _ in range(limit):
                logits=backend.model.logits(mx.array([ids],dtype=mx.int32))[0]
                token=int(mx.argmax(logits[0,-1,:]).item())
                if token==backend.eos:stop='end';break
                ids.append(token);output.append(token)
            if cached['token_ids']!=output or cached['stop']!=stop:
                raise ValueError('Cached/full-context generation mismatch: '+repr(prompt[:100]))
            if any(i in (0,2,3) for i in output):raise ValueError('Unexpected structural token in generation check')
            checks.append(dict(prompt=prompt,**cached,cached_uncached_equal=True))
        if not any(r['tokens']>=3 for r in checks):raise ValueError('Generation check must exercise multiple continuation steps')
        return checks
    finally:backend.model.train(was_training)

