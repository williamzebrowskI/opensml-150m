"""Live HF rows -> RAM token buffers -> batches; no materialized corpus or shards."""

from collections import OrderedDict
import copy
import hashlib
import importlib.metadata
import queue
import threading
import time

import numpy as np

from .common import fingerprint
from .corpus import accepted, content_key, identity, partition


SHUFFLE_FORMAT = 'sml-v2-document-shuffle-v1'


def with_stream_settings(manifest, settings):
    result = copy.deepcopy(manifest)
    result['settings'] = copy.deepcopy(settings)
    result['document_order'] = ('bounded-accepted-document-shuffle-v1'
                                if settings.get('shuffle_buffer', 0)
                                else 'source-order-no-row-shuffle-v1')
    return result


def stream_manifest(corpus, tokenizer, settings):
    if identity(corpus) != tokenizer.manifest['corpus_fingerprint']:
        raise ValueError('Streaming corpus differs from the frozen tokenizer corpus')
    return with_stream_settings(dict(format='sml-v2-hf-stream-v1', corpus=corpus,
                tokenizer=tokenizer.fingerprint, settings=settings,
                weights={s['label']: s['weight'] for s in corpus['sources']},
                datasets=importlib.metadata.version('datasets'),
                document_order='source-order-no-row-shuffle-v1'), settings)


class StreamExhausted(RuntimeError):
    """Natural source exhaustion, distinct from a network or filtering failure."""


def empty_shuffle_state(manifest):
    return dict(format=SHUFFLE_FORMAT, capacity=manifest['settings']['shuffle_buffer'],
                sources={s: dict(documents=[], draws=0, exhausted=False)
                         for s in manifest['weights']})


def open_dataset(source):
    from datasets import load_dataset
    if source.get('local_jsonl'):
        # Local fixtures exercise the same HF streaming/checkpoint machinery.
        return load_dataset('json', data_files=source['local_jsonl'], split='train', streaming=True)
    return load_dataset(source['repo'], source.get('config'), revision=source['revision'],
                        split=source['split'], streaming=True)


def retryable(exc):
    import httpx
    import requests
    if isinstance(exc, (httpx.HTTPStatusError, requests.HTTPError)):
        response = getattr(exc, 'response', None)
        return response is not None and (response.status_code in (408, 429) or response.status_code >= 500)
    return isinstance(exc, (OSError, EOFError, httpx.TransportError, requests.ConnectionError,
                            requests.Timeout))


class SourceCursor:
    def __init__(self, source, attempts, state=None):
        self.source, self.attempts = source, attempts
        self.dataset = self.iterator = None
        self.position = copy.deepcopy(state['dataset']) if state else None
        self.rows = state['rows'] if state else 0
        self.buffer = list(state['buffer']) if state else []
        if type(self.rows) is not int or self.rows < 0 or (self.rows and self.position is None):
            raise ValueError('Invalid streaming source cursor')

    def close(self):
        if self.iterator is not None and hasattr(self.iterator, 'close'):
            self.iterator.close()
        self.dataset = self.iterator = None

    def next(self):
        for attempt in range(self.attempts):
            try:
                if self.iterator is None:
                    self.dataset = open_dataset(self.source)
                    if self.position is not None:
                        self.dataset.load_state_dict(copy.deepcopy(self.position))
                    self.iterator = iter(self.dataset)
                row = next(self.iterator)
                self.position = self.dataset.state_dict()
                self.rows += 1
                return row
            except StopIteration as exc:
                raise StreamExhausted(f'{self.source["label"]}: stream exhausted; refusing silent recycling') from exc
            except Exception as exc:
                if not retryable(exc) or attempt + 1 == self.attempts:
                    raise
                self.close()
                print(f'[stream-retry] {self.source["label"]}: row={self.rows} '
                      f'attempt={attempt + 2}/{self.attempts} ({type(exc).__name__})', flush=True)
                time.sleep(min(8, 2 ** attempt))

    def state_dict(self):
        return dict(dataset=copy.deepcopy(self.position), rows=self.rows, buffer=list(self.buffer))


class StreamingTokens:
    def __init__(self, manifest, tokenizer, split, seq_len, state=None, *, exclusion_keys=None):
        self.exclusion_keys = exclusion_keys
        self.fingerprint = fingerprint(manifest)
        self.manifest, self.tokenizer = manifest, tokenizer
        self.split, self.seq_len = split, seq_len
        self.weights = manifest['weights']
        self.config = manifest['corpus']
        if split not in ('train', 'validation') or seq_len < 1:
            raise ValueError('Invalid streaming split/context')
        self.offsets = {s: 0 for s in self.weights}
        self.seen = OrderedDict()
        if state is not None:
            if (state['fingerprint'], state['split'], state['seq_len']) != (self.fingerprint, split, seq_len):
                raise ValueError('Saved streaming cursor is incompatible')
            if set(state['offsets']) != set(self.weights) or set(state['sources']) != set(self.weights):
                raise ValueError('Saved streaming source mismatch')
            if any(type(n) is not int or n < 0 or n % seq_len for n in state['offsets'].values()):
                raise ValueError('Invalid streaming token offsets')
            seen = state['recent_document_hashes']
            if len(seen) > manifest['settings']['dedup_window'] or len(set(seen)) != len(seen):
                raise ValueError('Invalid streaming dedup window')
            self.offsets = dict(state['offsets'])
            self.seen = OrderedDict.fromkeys(seen)
        self.sources = {s['label']: SourceCursor(s, manifest['settings']['read_attempts'],
                       state['sources'][s['label']] if state else None) for s in self.config['sources']}
        capacity = manifest['settings'].get('shuffle_buffer', 0)
        if type(capacity) is not int or capacity < 0 or capacity > 10000 or capacity == 1:
            raise ValueError('Invalid document shuffle capacity')
        self.shuffle = None
        if split == 'train' and capacity:
            self.shuffle = copy.deepcopy(state.get('shuffle')) if state is not None else empty_shuffle_state(manifest)
            sh = self.shuffle
            if (not isinstance(sh, dict) or set(sh) != {'format', 'capacity', 'sources'}
                    or sh['format'] != SHUFFLE_FORMAT or sh['capacity'] != capacity
                    or not isinstance(sh['sources'], dict) or set(sh['sources']) != set(self.sources)):
                raise ValueError('Missing or incompatible saved shuffle state')
            for saved in sh['sources'].values():
                if (not isinstance(saved, dict) or set(saved) != {'documents', 'draws', 'exhausted'}
                        or type(saved['draws']) is not int or saved['draws'] < 0
                        or type(saved['exhausted']) is not bool
                        or not isinstance(saved['documents'], list)
                        or len(saved['documents']) > capacity):
                    raise ValueError('Invalid saved document shuffle state')
                for text in saved['documents']:
                    if (not isinstance(text, str) or len(text) < self.config['min_chars']
                            or len(text.encode('utf-8')) > self.config['max_bytes']
                            or partition(content_key(text)) != 'train'):
                        raise ValueError('Invalid document in shuffle checkpoint')
        elif state is not None and 'shuffle' in state:
            raise ValueError('Shuffle state cannot be discarded on resume')
        for cursor in self.sources.values():
            if len(cursor.buffer) > self.config['max_bytes'] + seq_len + 1:
                raise ValueError('Invalid streaming token buffer length')
            if any(type(t) is not int or t < 0 or t >= tokenizer.vocab_size for t in cursor.buffer):
                raise ValueError('Invalid token in streaming cursor')

    def close(self):
        for cursor in self.sources.values():
            cursor.close()

    def state_dict(self):
        state = dict(fingerprint=self.fingerprint, split=self.split, seq_len=self.seq_len,
                    offsets=dict(self.offsets), recent_document_hashes=list(self.seen),
                    sources={s: c.state_dict() for s, c in self.sources.items()})
        if self.shuffle is not None:
            state['shuffle'] = copy.deepcopy(self.shuffle)
        return state

    def _accepted_document(self, source):
        cursor = self.sources[source]
        while True:
            if cursor.rows >= self.config['max_rows_per_source']:
                raise RuntimeError(f'{source}: scan limit reached; no silent source substitution')
            raw = cursor.next()
            text = accepted(raw, cursor.source, self.config)
            if text is None:
                continue
            key = content_key(text)
            if partition(key) != self.split or key in self.seen:
                continue
            if self.exclusion_keys is not None:
                from .stage_b_data import document_keys
                self.exclusion_keys.update(document_keys(raw, text))
            self.seen[key] = None
            if len(self.seen) > self.manifest['settings']['dedup_window']:
                self.seen.popitem(last=False)
            return text

    def _document(self, source):
        if self.shuffle is None:
            return self._accepted_document(source)
        state = self.shuffle['sources'][source]
        documents = state['documents']
        while len(documents) < self.shuffle['capacity'] and not state['exhausted']:
            try:
                documents.append(self._accepted_document(source))
            except StreamExhausted:
                state['exhausted'] = True
        if not documents:
            raise StreamExhausted(f'{source}: shuffled stream exhausted; refusing silent recycling')
        # Counter-based draws need only an integer in the checkpoint. Unlike HF's
        # shuffle iterator, all unconsumed documents are explicitly restored.
        draw = f'{self.config["seed"]}:{source}:{state["draws"]}'.encode()
        index = int.from_bytes(hashlib.sha256(draw).digest(), 'big') % len(documents)
        state['draws'] += 1
        text = documents[index]
        documents[index] = documents[-1]
        documents.pop()
        return text

    def row(self, source=None):
        source = source or min(self.weights, key=lambda s: self.offsets[s] / self.weights[s])
        cursor = self.sources[source]
        while len(cursor.buffer) < self.seq_len + 1:
            ids = self.tokenizer.encode(self._document(source)) + [self.tokenizer.eos]
            cursor.buffer.extend(ids)
        row = np.asarray(cursor.buffer[:self.seq_len + 1], dtype=np.int32)
        del cursor.buffer[:self.seq_len]
        self.offsets[source] += self.seq_len
        return row[:-1], row[1:]

    def batch(self, count, source=None):
        rows = [self.row(source) for _ in range(count)]
        return np.stack([r[0] for r in rows]), np.stack([r[1] for r in rows])

    def global_batch(self, accum, batches):
        pairs = [self.batch(sum(batches)) for _ in range(accum)]
        return np.stack([[p[axis] for p in pairs] for axis in (0, 1)])


class PrefetchedStream:
    """Checkpoint the consumed cursor, not the producer's ahead-of-training cursor."""

    def __init__(self, stream, depth):
        self.stream, self.depth = stream, depth
        self.consumed = stream.state_dict()
        self.closed = threading.Event()
        self.queue = queue.Queue(maxsize=max(1, depth))
        self.worker = self.geometry = None

    def _put(self, value):
        while not self.closed.is_set():
            try:
                self.queue.put(value, timeout=.1)
                return
            except queue.Full:
                pass

    def _run(self, accum, batches):
        try:
            while not self.closed.is_set():
                value = self.stream.global_batch(accum, batches)
                self._put((value, self.stream.state_dict(), None))
        except Exception as exc:
            self._put((None, None, exc))
        finally:
            self.stream.close()

    def global_batch(self, accum, batches):
        geometry = (accum, tuple(batches))
        if self.geometry is not None and self.geometry != geometry:
            raise ValueError('Prefetch geometry cannot change within a run')
        self.geometry = geometry
        if self.depth == 0:
            value = self.stream.global_batch(accum, batches)
            self.consumed = self.stream.state_dict()
            return value
        if self.worker is None:
            self.worker = threading.Thread(target=self._run, args=(accum, batches), daemon=True,
                                           name='v1-hf-prefetch')
            self.worker.start()
        while not self.closed.is_set():
            try:
                value, state, error = self.queue.get(timeout=.1)
                if error is not None:
                    raise RuntimeError(f'Live stream failed: {error}') from error
                self.consumed = state
                return value
            except queue.Empty:
                if not self.worker.is_alive():
                    raise RuntimeError('Streaming worker exited without a batch')
        raise RuntimeError('Stream is closed')

    def state_dict(self):
        return copy.deepcopy(self.consumed)

    def close(self):
        self.closed.set()
        if self.worker is not None:
            self.worker.join(timeout=2)
        else:
            self.stream.close()


def validation_batches(manifest, tokenizer, seq_len, count, batch_size, *, exclusion_keys=None):
    stream = StreamingTokens(manifest, tokenizer, 'validation', seq_len, exclusion_keys=exclusion_keys)
    batches, digest = {}, hashlib.sha256()
    try:
        for source in manifest['weights']:
            print(f'[eval-stream] collecting fixed held-out batches in RAM: {source}', flush=True)
            batches[source] = []
            for index in range(count):
                batches[source].append(stream.batch(batch_size, source))
                if count > 16 and (index + 1) % 16 == 0:
                    print(f'[eval-stream] {source}: {index + 1}/{count} batches ready', flush=True)
            digest.update(source.encode())
            for x, y in batches[source]:
                digest.update(x.astype('<i4').tobytes())
                digest.update(y.astype('<i4').tobytes())
    finally:
        stream.close()
    return batches, digest.hexdigest()
