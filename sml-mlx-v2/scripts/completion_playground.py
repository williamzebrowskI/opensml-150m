#!/usr/bin/env python3
"""Local streaming completion playground; no saved conversation history."""

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field, ConfigDict, model_validator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
V2_ROOT = ROOT
sys.path.insert(0, str(V2_ROOT))
DEFAULT_RUN = ROOT / 'runs/stage_b_prose_v1'
DEFAULT_V2_RUN = V2_ROOT / 'runs/full_15b_five_mac_shuffled_v1'
DEFAULT_V2_TOKENIZER = V2_ROOT / 'tokenizer/bytebpe32k_v1'
ASSETS = ROOT / 'playground'
SFT_PREVIEW_LOCK = threading.Lock()
PLAIN_SFT_FORMAT = 'plain-user-assistant-eos-v1'
SFT_FORMATS = ('opensml-sft-v1', PLAIN_SFT_FORMAT)


class NoSFTPreview(FileNotFoundError):
    pass


class Message(BaseModel):
    model_config = ConfigDict(extra='forbid')
    role: str = Field(pattern='^(user|assistant)$')
    content: str = Field(max_length=12000)


class GenerationRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    model_id: str | None = Field(default=None, pattern='^[a-z0-9_-]{1,48}$')
    messages: list[Message] = Field(min_length=1, max_length=100)
    instruction: str = Field(default='', max_length=12000)
    max_tokens: int = Field(default=64, ge=1, le=2048, strict=True)
    temperature: float = Field(default=0.7, ge=0, le=2, allow_inf_nan=False)
    top_k: int = Field(default=40, ge=0, le=1000, strict=True)
    repetition_penalty: float = Field(default=1.1, ge=1, le=2, allow_inf_nan=False)
    seed: int = Field(default=2026, ge=0, le=2**32-1, strict=True)

    @model_validator(mode='after')
    def check_messages(self):
        if self.messages[-1].role != 'user' or not self.messages[-1].content.strip():
            raise ValueError('Finish with a nonempty user prompt')
        if len(self.instruction) + sum(len(m.content) for m in self.messages) > 32000:
            raise ValueError('Conversation is too long; clear it and start a new prompt')
        return self


def completion_text(messages):
    # A pretrained base model has no instruction/chat template. Assistant text
    # continues its preceding prompt; a follow-up starts a new paragraph.
    text = ''
    for message in messages:
        if message.role == 'user' and text:
            text += '\n\n'
        text += message.content
    return text


def penalize_repetition(logits, generated, penalty):
    """Reduce scores for distinct tokens in the last 64 response tokens.

    Apply the sign-aware repetition penalty before temperature/top-k selection.
    Prompt tokens are excluded; 1.0 preserves the original sampling path.
    """
    if penalty == 1.0 or not generated:
        return logits
    import mlx.core as mx

    tokens = mx.array(sorted(set(generated[-64:])), dtype=mx.int32)[None, :]
    scores = logits.astype(mx.float32)
    selected = mx.take_along_axis(scores, tokens, axis=-1)
    adjusted = mx.where(selected < 0, selected * penalty, selected / penalty)
    return mx.put_along_axis(scores, tokens, adjusted, axis=-1)


def prepare_prompt(tokenizer, options, context_limit, add_bos=False,
                   preserve_prompt=False, output_limit=192):
    """Keep an optional instruction intact while trimming older conversation."""
    instruction = options.instruction.strip()
    prefix = tokenizer.encode(instruction + '\n\n', out_type=int) if instruction else []
    instruction_tokens = len(prefix)
    if add_bos and tokenizer.bos_id() >= 0:
        prefix.insert(0, tokenizer.bos_id())
    ids = tokenizer.encode(completion_text(options.messages), out_type=int)
    if not ids:
        raise ValueError('Prompt has no tokens')
    max_new = min(options.max_tokens, output_limit, context_limit - 1)
    if preserve_prompt:
        # Keep the prompt before budgeting V2 output; retain at least one new token.
        remaining = context_limit - len(prefix) - 1
        if remaining < 1:
            raise ValueError('Instruction is too long. Shorten it to leave room for your message and output.')
        trimmed = max(0, len(ids) - remaining)
        prompt = prefix + ids[-remaining:]
        return dict(ids=prompt, instruction_tokens=instruction_tokens,
                    trimmed_tokens=trimmed, max_tokens=min(max_new, context_limit - len(prompt)))
    remaining = context_limit - max_new - len(prefix)
    if remaining < 1:
        raise ValueError('Instruction is too long for the selected output limit. '
                         'Shorten it or lower Max new tokens to leave room for your message.')
    trimmed = max(0, len(ids) - remaining)
    return dict(ids=prefix + ids[-remaining:], instruction_tokens=instruction_tokens,
                trimmed_tokens=trimmed, max_tokens=max_new)


def prepare_sft_prompt(tokenizer, options, context_limit):
    """Match opensml-sft-v1: explicit turn IDs and separately encoded role/body.

    Keep the system turn and newest user turn intact. Drop complete oldest
    user/assistant pairs when necessary, never slice through a role header.
    """
    for i, message in enumerate(options.messages):
        if message.role != ('user' if i % 2 == 0 else 'assistant'):
            raise ValueError('SFT conversation must alternate user and assistant turns')

    def turn(role, content):
        return [2] + tokenizer.encode(role + '\n') + tokenizer.encode(content) + [3]

    instruction = options.instruction.strip()
    prefix = turn('system', instruction) if instruction else []
    turns = [turn(message.role, message.content) for message in options.messages]
    suffix = [2] + tokenizer.encode('assistant\n')
    size = len(prefix) + sum(map(len, turns)) + len(suffix)
    trimmed = 0
    while size >= context_limit and len(turns) > 1:
        removed = len(turns[0]) + len(turns[1])
        turns = turns[2:]
        size -= removed
        trimmed += removed
    if size >= context_limit:
        raise ValueError('System instruction and latest message are too long. '
                         'Shorten them to leave room for an answer.')
    return dict(ids=prefix + [token for turn_ids in turns for token in turn_ids] + suffix,
                instruction_tokens=len(prefix), trimmed_tokens=trimmed,
                max_tokens=min(options.max_tokens, 2048, context_limit - size))


def prepare_plain_sft_prompt(tokenizer, options, context_limit):
    """Put optional context inside the single user turn this format expects."""
    for i, message in enumerate(options.messages):
        if message.role != ('user' if i % 2 == 0 else 'assistant'):
            raise ValueError('SFT conversation must alternate user and assistant turns')

    instruction = options.instruction.strip()
    history = options.messages[:-1]
    latest = options.messages[-1].content

    def encode(messages):
        # The no-context path must stay byte-for-byte identical to training.
        if not instruction and not messages:
            return tokenizer.encode(f'User: {latest}\nAssistant:')
        parts = []
        if instruction:
            parts.append(f'Instruction: {instruction}')
        if messages:
            transcript = '\n'.join(f'{message.role.title()}: {message.content}' for message in messages)
            parts.append(f'Previous messages:\n{transcript}')
        parts.append(f'Current message: {latest}')
        content = '\n\n'.join(parts)
        return tokenizer.encode(f'User: {content}\nAssistant:')

    ids = encode(history)
    trimmed = 0
    while len(ids) >= context_limit and history:
        history = history[2:]
        new_ids = encode(history)
        trimmed += len(ids) - len(new_ids)
        ids = new_ids
    if len(ids) >= context_limit:
        raise ValueError('Instruction and current message are too long. Shorten them to leave room for an answer.')
    return dict(ids=ids, instruction_tokens=len(tokenizer.encode(instruction)) if instruction else 0,
                trimmed_tokens=trimmed,
                max_tokens=min(options.max_tokens, 2048, context_limit-len(ids)))


@dataclass
class Checkpoint:
    bundle: Path
    metadata: dict
    manifest: dict
    selection: str = 'best'

    @property
    def model_type(self):
        return 'sft' if self.metadata.get('training_format') in SFT_FORMATS else 'base'

    @property
    def family(self):
        if self.model_type != 'sft' and 'v2_contract' not in self.metadata:
            raise ValueError('Only V2 checkpoint metadata is supported')
        return 'v2'

    @property
    def config(self):
        if self.model_type == 'sft':
            return self.metadata['model']
        return self.metadata['recipe']['model']

    def public(self):
        meta = self.metadata
        report = meta.get('last_report') or {}
        evaluated = self.model_type == 'sft' and report.get('step') == meta['step']
        quality = None
        if evaluated:
            quality = dict(passed=report.get('eligible'),
                           content_accuracy=report.get('generation', {}).get('accuracy'),
                           stop_rate=report.get('generation', {}).get('stop_rate'),
                           retention_increase=report.get('retention_increase'))
        validation_loss = report['assistant_loss'] if evaluated else meta.get('best_val_loss')
        validation_kind = 'assistant-only SFT' if self.model_type == 'sft' else 'pretraining'
        # Pilot exports contain a natural-instruction evaluation, not the full
        # trainer's assistant loss or cumulative target counter.
        evaluation = meta.get('evaluation') or {}
        if self.model_type == 'sft' and evaluation.get('step') == meta['step']:
            validation_loss = evaluation.get('natural_nll', {}).get('loss')
            validation_kind = 'natural-instruction'
        return dict(checkpoint=self.bundle.name, step=meta['step'],
                    training_format=meta.get('training_format'),
                    single_turn_only=meta.get('training_format') == PLAIN_SFT_FORMAT,
                    tokens=meta.get('tokens'),
                    model_type=self.model_type, family=self.family,
                    token_kind='assistant targets' if self.model_type == 'sft' else 'pretraining tokens',
                    validation_kind=validation_kind,
                    tokenizer='Own 32K byte-BPE',
                    validation_loss=validation_loss,
                    selection=self.selection, quality=quality, context_limit=self.config['max_seq_len'],
                    max_output_tokens=min(2048, self.config['max_seq_len']),
                    ffn_impl=self.config.get('ffn_impl', 'reference'))


@dataclass(frozen=True)
class ModelSource:
    model_id: str
    label: str
    run: Path
    family: str
    tokenizer_dir: Path | None = None
    model_type: str = 'base'
    selection: str = 'best'

    def checkpoint(self):
        if self.selection == 'pinned':
            checkpoint = read_pinned_sft(self.run)
        else:
            checkpoint = read_best_loss(self.run) if self.selection == 'best-loss' else read_best(self.run)
        if checkpoint.family != self.family:
            raise ValueError('Checkpoint architecture does not match the selected model')
        if checkpoint.model_type != self.model_type:
            raise ValueError('Checkpoint training format does not match the selected model')
        return checkpoint


class V2TokenizerAdapter:
    """Expose the playground interface without changing V2's text encoding."""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.fingerprint = tokenizer.fingerprint

    def encode(self, text, out_type=int):
        return self.tokenizer.encode(text)

    def decode(self, ids):
        return self.tokenizer.decode(ids)

    def bos_id(self):
        return -1

    def eos_id(self):
        return self.tokenizer.eos


def checkpoint_tokenizer(checkpoint, tokenizer_dir=None):
    meta = checkpoint.metadata
    if checkpoint.family == 'v2':
        from sml_v2.tokenization import Tokenizer
        tokenizer = Tokenizer(tokenizer_dir or DEFAULT_V2_TOKENIZER)
        fingerprint = meta['tokenizer'] if checkpoint.model_type == 'sft' else meta['v2_contract']['tokenizer']
        if tokenizer.fingerprint != fingerprint:
            raise ValueError('Tokenizer fingerprint mismatch')
        if tokenizer.vocab_size != checkpoint.config['vocab_size']:
            raise ValueError('Tokenizer vocabulary does not match the model')
        return V2TokenizerAdapter(tokenizer)
    raise ValueError('Only V2 checkpoints are supported')


def read_best(run):
    run = Path(run).resolve()
    pointer = json.loads((run / 'best.json').read_text())
    name = pointer.get('bundle', '')
    if not isinstance(name, str) or not re.fullmatch(r'step_\d+_[a-f0-9]+', name):
        raise ValueError('Invalid best checkpoint pointer')
    bundle = run / name
    return read_checkpoint(bundle)


def read_checkpoint(bundle):
    if bundle.is_symlink() or not bundle.is_dir():
        raise ValueError('Best checkpoint bundle is unavailable')
    manifest_path = bundle / 'manifest.json'
    meta_path = bundle / 'model.safetensors.json'
    if manifest_path.is_symlink() or meta_path.is_symlink():
        raise ValueError('Checkpoint files must not be symlinks')
    manifest = json.loads(manifest_path.read_text())
    payload = meta_path.read_bytes()
    record = manifest['files']['model.safetensors.json']
    if len(payload) != record['bytes'] or hashlib.sha256(payload).hexdigest() != record['sha256']:
        raise ValueError('Checkpoint metadata checksum mismatch')
    metadata = json.loads(payload)
    if metadata['step'] != manifest['step']:
        raise ValueError('Checkpoint step mismatch')
    if metadata.get('sft') or metadata.get('training_format'):
        if metadata.get('sft') is not True or metadata.get('training_format') not in SFT_FORMATS:
            raise ValueError('Unsupported SFT checkpoint training format')
    return Checkpoint(bundle, metadata, manifest)


def read_pinned_sft(path):
    """Read a saved bundle or inference-only export without changing its files."""
    path = Path(path)
    if path.is_symlink():
        raise ValueError('Saved SFT checkpoint must not be a symlink')
    if path.is_dir():
        checkpoint = read_checkpoint(path)
    else:
        if path.name != 'model.safetensors' or not path.is_file():
            raise ValueError('Expected an SFT bundle or model.safetensors export')
        meta_path = path.with_name(path.name + '.json')
        if meta_path.is_symlink():
            raise ValueError('Checkpoint metadata must not be a symlink')
        payload = meta_path.read_bytes()
        meta = json.loads(payload)
        if (meta.get('sft') is not True or meta.get('training_format') not in SFT_FORMATS
                or meta.get('experiment') is not True):
            raise ValueError('Unsupported SFT inference export')
        digest = meta.get('weights_sha256', '')
        if not isinstance(digest, str) or not re.fullmatch(r'[a-f0-9]{64}', digest):
            raise ValueError('SFT export requires a valid weights checksum')
        manifest = dict(step=meta['step'], files={
            'model.safetensors': dict(bytes=path.stat().st_size, sha256=digest),
            'model.safetensors.json': dict(bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest()),
        })
        checkpoint = Checkpoint(path.parent, meta, manifest)
    if checkpoint.model_type != 'sft':
        raise ValueError('Saved checkpoint must use the SFT training format')
    checkpoint.selection = 'pinned'
    return checkpoint


def read_best_loss(run):
    """Pin the lowest-loss retained evaluated SFT bundle outside trainer pruning.

    Training metadata, quality eligibility and trainer pointers remain unchanged.
    The complete copied bundle preserves optimizer/cursor recovery as well as
    preview weights. A pruned historical checkpoint cannot be reconstructed.
    """
    from sml_v2.checkpoint_metadata import verify_bundle_metadata
    from sml_v2.common import atomic_json
    run = Path(run).resolve()
    archive = run / 'playground_best_loss'
    with SFT_PREVIEW_LOCK:
        candidates = []
        if (archive / 'best.json').exists():
            candidates.append(read_best(archive))
        for path in run.glob('step_*_*'):
            try:
                checkpoint = read_checkpoint(path)
                meta = checkpoint.metadata
                report = meta.get('last_report') or {}
                loss = report.get('assistant_loss')
                if (checkpoint.model_type == 'sft' and meta['step'] > 0
                        and report.get('step') == meta['step']
                        and isinstance(loss, (int, float)) and math.isfinite(loss)):
                    candidates.append(checkpoint)
            except FileNotFoundError:
                continue  # A concurrent checkpoint rotation removed this bundle.
        if not candidates:
            raise NoSFTPreview('Waiting for the first evaluated SFT checkpoint. This page checks automatically.')
        chosen = min(candidates, key=lambda c: (c.metadata['last_report']['assistant_loss'], c.metadata['step']))
        if chosen.bundle.parent != archive:
            archive.mkdir(exist_ok=True)
            target = archive / chosen.bundle.name
            if not target.exists():
                temporary = archive / ('.copy-' + uuid.uuid4().hex)
                try:
                    if sys.platform == 'darwin':
                        subprocess.run(['/bin/cp', '-cR', str(chosen.bundle), str(temporary)],
                                       capture_output=True, text=True, check=True, timeout=60)
                    else:
                        shutil.copytree(chosen.bundle, temporary, symlinks=True)
                    verify_bundle_metadata(temporary)
                    os.replace(temporary, target)
                finally:
                    if temporary.exists():
                        shutil.rmtree(temporary)
            chosen = read_checkpoint(target)
            atomic_json(archive / 'best.json', dict(format='sml-pretrain-bundle-v1', bundle=target.name,
                        selection='lowest-retained-assistant-validation-loss',
                        loss=chosen.metadata['last_report']['assistant_loss'],
                        quality_passed=chosen.metadata['last_report']['eligible']))
        chosen.selection = 'best-loss'
        return chosen


class InferenceEngine:
    """All MLX operations run on one dedicated worker thread."""
    def __init__(self, run=DEFAULT_RUN, sources=None, default_model='v2'):
        self.run = Path(run)
        self.sources = sources or {'v2': ModelSource('v2', 'V2 pretrained - best', self.run, 'v2')}
        self.default_model = default_model
        self.model = self.tokenizer = self.identity = None

    def load(self, checkpoint, tokenizer_dir=None):
        import mlx.core as mx
        from sml_v2.model import TransformerConfig, TransformerLM
        from sml_v2.inference import _cast_model_floats

        identity = (str(checkpoint.bundle.resolve()), checkpoint.family,
                    checkpoint.manifest['files']['model.safetensors']['sha256'],
                    checkpoint.manifest['files']['model.safetensors.json']['sha256'],
                    str(tokenizer_dir))
        if self.identity == identity:
            return
        # Release the previous model before constructing the next one.
        self.model = self.tokenizer = self.identity = None
        mx.clear_cache()
        mx.set_default_device(mx.gpu)
        mx.set_cache_limit(64 * 1024**2)
        mx.reset_peak_memory()
        limit = 2048 if checkpoint.family == 'v2' else 256
        if not 1 < checkpoint.config['max_seq_len'] <= limit:
            raise ValueError('Checkpoint exceeds the supported context limit')
        tokenizer = checkpoint_tokenizer(checkpoint, tokenizer_dir)
        record = checkpoint.manifest['files']['model.safetensors']
        budget_mib = 800 if checkpoint.metadata.get('compute_precision') == 'float32' else 400
        if record['bytes'] > budget_mib * 1024**2:
            raise ValueError('Checkpoint exceeds this playground’s model-weight budget')
        weights = checkpoint.bundle / 'model.safetensors'
        if weights.is_symlink():
            raise ValueError('Weights must not be a symlink')
        # A private temporary model-only copy survives concurrent checkpoint
        # retention. No optimizer or data cursor is loaded. The copy is removed
        # as soon as MLX finishes materializing the weights.
        with tempfile.TemporaryDirectory(prefix='sml-playground-') as tmp:
            target = Path(tmp) / 'model.safetensors'
            digest = hashlib.sha256()
            size = 0
            with weights.open('rb') as src, target.open('wb') as dst:
                for block in iter(lambda: src.read(4 * 1024**2), b''):
                    dst.write(block)
                    digest.update(block)
                    size += len(block)
            if digest.hexdigest() != record['sha256'] or size != record['bytes']:
                raise ValueError('Model weights checksum mismatch')
            model = TransformerLM(TransformerConfig(**checkpoint.config))
            precision = mx.float32 if checkpoint.metadata.get('compute_precision') == 'float32' else mx.bfloat16
            _cast_model_floats(model, precision)
            model.load_weights(str(target), strict=True)
            model.eval()
            mx.eval(model.parameters())
        self.model, self.tokenizer, self.identity = model, tokenizer, identity

    def generate(self, options, emit, cancelled):
        import mlx.core as mx
        import numpy as np
        from sml_v2.inference import _sample_next_id

        source = self.sources[options.model_id or self.default_model]
        checkpoint = None
        for attempt in range(3):
            if cancelled.is_set():
                return
            try:
                checkpoint = source.checkpoint()
                emit(dict(type='status', message=f'Loading best checkpoint · step {checkpoint.metadata["step"]:,}'))
                self.load(checkpoint, source.tokenizer_dir)
                break
            except FileNotFoundError:
                if attempt == 2:
                    raise ValueError('Best checkpoint changed during loading; please retry')
        model, tokenizer = self.model, self.tokenizer
        if checkpoint.metadata.get('training_format') == PLAIN_SFT_FORMAT:
            prepared = prepare_plain_sft_prompt(tokenizer, options, model.cfg.max_seq_len)
        elif checkpoint.model_type == 'sft':
            prepared = prepare_sft_prompt(tokenizer, options, model.cfg.max_seq_len)
        else:
            prepared = prepare_prompt(tokenizer, options, model.cfg.max_seq_len,
                                      checkpoint.metadata.get('args', {}).get('add_bos', False),
                                      preserve_prompt=checkpoint.family == 'v2',
                                      output_limit=2048 if checkpoint.family == 'v2' else 192)
        ids, max_new = prepared['ids'], prepared['max_tokens']
        emit(dict(type='start', **checkpoint.public(), model_id=source.model_id,
                  model_label=source.label, run_name=source.run.name, prompt_tokens=len(ids),
                  trimmed_tokens=prepared['trimmed_tokens'], max_tokens=max_new,
                  instruction_tokens=prepared['instruction_tokens'],
                  repetition_penalty=options.repetition_penalty))
        full_context = checkpoint.metadata.get('decode_mode') == 'full_context'
        generated, caches = [], None
        start = time.perf_counter()
        try:
            if cancelled.is_set():
                return
            logits, caches = model.logits(mx.array([ids], dtype=mx.int32))
            scores = logits[:, -1, :]
            rng = np.random.default_rng(options.seed)
            reason = 'length'
            for _ in range(max_new):
                if cancelled.is_set():
                    reason = 'stopped'
                    break
                sampling_scores = penalize_repetition(scores, generated, options.repetition_penalty)
                token = _sample_next_id(sampling_scores, options.temperature, options.top_k, rng)
                if token == tokenizer.eos_id():
                    reason = 'eos'
                    break
                if checkpoint.model_type == 'sft' and token in (0, 2, 3):
                    reason = 'end_turn' if token == 3 and checkpoint.metadata.get('training_format') != PLAIN_SFT_FORMAT else 'special_token'
                    break
                generated.append(token)
                text = tokenizer.decode(generated)
                # Send the complete decoded prefix so Unicode token boundaries
                # can be corrected without accumulating garbled text in the UI.
                elapsed = time.perf_counter()-start
                emit(dict(type='text', text=text, generated_tokens=len(generated),
                          seconds=elapsed, tokens_per_second=len(generated)/max(elapsed, 1e-6)))
                if len(generated) < max_new and not cancelled.is_set():
                    if full_context:
                        logits, _ = model.logits(mx.array([ids + generated], dtype=mx.int32))
                        scores = logits[:, -1, :]
                    else:
                        scores, caches = model.step(mx.array([[token]], dtype=mx.int32), caches=caches)
            elapsed = time.perf_counter()-start
            emit(dict(type='done', text=tokenizer.decode(generated), generated_tokens=len(generated),
                      seconds=elapsed, tokens_per_second=len(generated)/max(elapsed, 1e-6),
                      stop_reason=reason, peak_gib=mx.get_peak_memory()/1024**3))
        finally:
            # Only model weights persist between requests. No conversation or KV
            # cache is retained on the server.
            caches = None
            mx.clear_cache()


class GenerationService:
    def __init__(self, engine):
        self.engine = engine
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='sml-inference')
        self.busy = threading.Lock()
        self.cancelled = None
        self.follow_stop = threading.Event()
        self.follow_thread = None

    def follow_best_loss(self, sources):
        followed = [source for source in sources.values() if source.selection == 'best-loss']
        if not followed:
            return
        def follow():
            previous_error = None
            while not self.follow_stop.is_set():
                for source in followed:
                    try:
                        source.checkpoint()
                        previous_error = None
                    except NoSFTPreview:
                        pass
                    except Exception as exc:
                        if str(exc) != previous_error:
                            print(f'[sft-preview] Could not preserve checkpoint: {exc}', flush=True)
                            previous_error = str(exc)
                self.follow_stop.wait(15)
        self.follow_thread = threading.Thread(target=follow, daemon=True, name='sft-best-loss')
        self.follow_thread.start()

    def start(self, options):
        if not self.busy.acquire(blocking=False):
            raise HTTPException(409, 'A generation is already running. Stop it or wait a moment.')
        events = queue.Queue(maxsize=32)
        cancelled, finished = threading.Event(), threading.Event()
        self.cancelled = cancelled

        def emit(event):
            while not cancelled.is_set():
                try:
                    events.put(event, timeout=0.1)
                    return
                except queue.Full:
                    pass

        def work():
            try:
                self.engine.generate(options, emit, cancelled)
            except Exception as exc:
                emit(dict(type='error', message=str(exc)))
            finally:
                finished.set()
                self.busy.release()
        try:
            self.pool.submit(work)
        except Exception:
            self.busy.release()
            raise
        return events, cancelled, finished

    def close(self):
        self.follow_stop.set()
        if self.follow_thread is not None:
            self.follow_thread.join(timeout=10)
        if self.cancelled is not None:
            self.cancelled.set()
        self.pool.shutdown(wait=False, cancel_futures=True)


def create_app(run=DEFAULT_RUN, engine=None, sources=None, default_model='v2'):
    app = FastAPI(docs_url=None, redoc_url=None)
    sources = sources or {'v2': ModelSource('v2', 'V2 pretrained - best', Path(run), 'v2')}
    if default_model not in sources:
        raise ValueError('Default model is not configured')
    service = GenerationService(engine or InferenceEngine(run, sources, default_model))
    service.follow_best_loss(sources)
    app.state.service = service

    @app.middleware('http')
    async def local_only(request, call_next):
        host = request.headers.get('host', '').split(':')[0]
        origin = request.headers.get('origin')
        if host not in ('127.0.0.1', 'localhost', 'testserver') or (
                origin and origin != str(request.base_url).rstrip('/')):
            from fastapi.responses import JSONResponse
            return JSONResponse({'detail': 'Local, same-origin requests only'}, status_code=403)
        try:
            content_length = int(request.headers.get('content-length', '0'))
        except ValueError:
            from fastapi.responses import JSONResponse
            return JSONResponse({'detail': 'Invalid content length'}, status_code=400)
        if content_length > 150000:
            from fastapi.responses import JSONResponse
            return JSONResponse({'detail': 'Request too large'}, status_code=413)
        response = await call_next(request)
        response.headers['Cache-Control'] = 'no-store'
        response.headers['X-Content-Type-Options'] = 'nosniff'
        return response

    @app.get('/')
    def index():
        return FileResponse(ASSETS / 'index.html')

    @app.get('/app.js')
    def javascript():
        return FileResponse(ASSETS / 'app.js', media_type='text/javascript')

    @app.get('/style.css')
    def stylesheet():
        return FileResponse(ASSETS / 'style.css', media_type='text/css')

    def select_source(model_id):
        if (model_id or default_model) not in sources:
            raise HTTPException(404, 'Unknown model')
        return sources[model_id or default_model]

    def source_status(source):
        info = dict(model_id=source.model_id, model_label=source.label, run_name=source.run.name)
        if source.model_type == 'sft' and source.selection == 'best' and not (source.run / 'best.json').exists():
            return dict(available=False, waiting_for_best=True, model_type='sft', **info,
                        message='Waiting for this SFT run to save its first best checkpoint. This page checks automatically.')
        try:
            return dict(available=True, **info, **source.checkpoint().public())
        except NoSFTPreview as exc:
            return dict(available=False, waiting_for_best=True, model_type='sft', **info, message=str(exc))
        except (OSError, ValueError, KeyError) as exc:
            return dict(available=False, **info, message=str(exc))

    @app.get('/api/models')
    def models():
        return dict(default_model=default_model,
                    models=[source_status(source) for source in sources.values()])

    @app.get('/api/status')
    def status(model_id: str | None = None):
        return dict(busy=service.busy.locked(), **source_status(select_source(model_id)))

    @app.post('/api/generate')
    async def generate(options: GenerationRequest, request: Request):
        state = source_status(select_source(options.model_id))
        if state.get('waiting_for_best'):
            raise HTTPException(503, state['message'])
        events, cancelled, finished = service.start(options)

        async def stream():
            try:
                while True:
                    if await request.is_disconnected():
                        break
                    try:
                        event = events.get_nowait()
                    except queue.Empty:
                        if finished.is_set():
                            break
                        await asyncio.sleep(0.015)
                        continue
                    yield 'data: ' + json.dumps(event, ensure_ascii=False) + '\n\n'
            finally:
                cancelled.set()
        return StreamingResponse(stream(), media_type='text/event-stream',
                                 headers={'X-Accel-Buffering': 'no'})

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--v2-run-dir', type=Path, default=DEFAULT_V2_RUN)
    parser.add_argument('--v2-tokenizer-dir', type=Path, default=DEFAULT_V2_TOKENIZER)
    parser.add_argument('--sft-run-dir', type=Path, help='Optionally register an existing SFT run')
    parser.add_argument('--sft-label', help='Display name for the optional SFT model')
    parser.add_argument('--sft-tokenizer-dir', type=Path, default=DEFAULT_V2_TOKENIZER)
    parser.add_argument('--sft-selection', choices=['best', 'best-loss'], default='best',
                        help='best: follow the training recipe\'s best pointer; best-loss: preserve lowest-loss available evaluated checkpoint')
    parser.add_argument('--sft-checkpoint', nargs=3, action='append', default=[],
                        metavar=('MODEL_ID', 'LABEL', 'PATH'),
                        help='Add a saved SFT bundle or inference export; repeat for multiple checkpoints')
    parser.add_argument('--default-model', default='v2', help='ID of a registered model')
    parser.add_argument('--port', type=int, default=8767)
    args = parser.parse_args()
    import uvicorn
    sources = {
        'v2': ModelSource('v2', 'V2 pretrained - best', args.v2_run_dir.resolve(),
                          'v2', args.v2_tokenizer_dir.resolve()),
    }
    if args.sft_run_dir is not None:
        label = args.sft_label or ('V2 SFT - best available loss' if args.sft_selection == 'best-loss' else 'V2 SFT - best')
        sources['v2-sft'] = ModelSource('v2-sft', label, args.sft_run_dir.resolve(),
                                        'v2', args.sft_tokenizer_dir.resolve(), model_type='sft', selection=args.sft_selection)
    for model_id, label, path in args.sft_checkpoint:
        if not re.fullmatch(r'[a-z0-9_-]{1,48}', model_id) or model_id in sources:
            parser.error('Saved SFT model IDs must be unique lowercase IDs')
        sources[model_id] = ModelSource(model_id, label, Path(path).absolute(), 'v2',
                                       args.sft_tokenizer_dir.resolve(), model_type='sft', selection='pinned')
    if args.default_model not in sources:
        parser.error('Default model must be a registered model ID')
    app = create_app(args.run_dir.resolve(), sources=sources, default_model=args.default_model)
    print(f'OpenSML playground: http://127.0.0.1:{args.port} (no saved chat history)', flush=True)
    try:
        uvicorn.run(app, host='127.0.0.1', port=args.port, access_log=False)
    finally:
        app.state.service.close()


if __name__ == '__main__':
    main()
