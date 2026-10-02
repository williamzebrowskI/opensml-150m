"""Token-based continuation accounting, independent of MLX and host count."""

import re
from copy import deepcopy

FORMAT = 'sml-jaccl-production-v1'
BATCHES = [32, 24, 24, 24]
TOKENS_PER_UPDATE = sum(BATCHES) * 2 * 256
SUPPORTED_BATCHES = ((32, 24, 24, 24), (32, 28, 28, 28))
EVENT_DEFAULTS = dict(eval_every_updates=500, sample_every_updates=1000,
                      comparison_every_updates=1000)


def add_target_argument(parser):
    parser.add_argument('--target-tokens', type=int,
                        help='New cumulative token target when resuming a stopped run; '
                             'preserves the original LR schedule and rounds up to a full update')


def extend_token_target(metadata, target=None):
    """Extend only the stopping budget; never rebase the LR schedule or cursor."""
    settings = continuation(metadata)
    if target is None:
        return metadata
    if type(target) is not int or target <= 0:
        raise ValueError('target-tokens must be a positive integer')
    if (not metadata.get('fresh_training') or settings.get('schedule_units') != 'optimizer_updates'
            or any(metadata.get(k) for k in ('web_phase', 'prose_phase', 'lr_trial'))):
        raise ValueError('Token-target extension requires the fresh four-Mac production run')
    previous = metadata.get('token_extension', {})
    if target == settings['planned_tokens'] or target == previous.get('requested_tokens'):
        return metadata  # Repeating the same command on resume is idempotent.
    if target < settings['planned_tokens']:
        raise ValueError('target-tokens must exceed the saved token budget')
    # A durable checkpoint can precede the old budget after a manual stop.
    # The launcher lock and cluster preflight enforce that training is idle;
    # budget completion is unrelated to safe data/optimizer restoration.
    result = deepcopy(metadata)
    reference = settings['reference_tokens_per_update']
    # Align to the original update grid. Both supported accumulation counts
    # divide this grid for the fresh recipe; the original schedule stays intact.
    target_budget = ((target + reference - 1) // reference) * reference
    result['distributed_training']['planned_tokens'] = target_budget
    result['token_extension'] = dict(format='sml-token-extension-v1',
        original_planned_tokens=reference * metadata['args']['max_steps'],
        previous_planned_tokens=settings['planned_tokens'],
        requested_tokens=target, planned_tokens=target_budget)
    continuation(result)
    return result


def batch_layout(batches):
    if (not isinstance(batches, (list, tuple))
            or any(type(value) is not int for value in batches)
            or tuple(batches) not in SUPPORTED_BATCHES):
        raise ValueError('Production batches must be 32,24,24,24 or 32,28,28,28')
    return list(batches)


def parse_batches(value):
    return batch_layout([int(part) for part in value.split(',')])


def add_batch_argument(parser):
    parser.add_argument('--batches', type=parse_batches,
                        help='Per-rank microbatches: 32,24,24,24 or 32,28,28,28; '
                             'defaults to the saved split, or 32,24,24,24 on migration')


def accumulation_steps(value):
    if type(value) is not int or value not in (2, 4):
        raise ValueError('Production gradient accumulation must be 2 or 4')
    return value


def add_accumulation_argument(parser):
    parser.add_argument('--grad-accum', type=int, choices=(2, 4),
                        help='Microbatches per optimizer update; defaults to the saved '
                             'accumulation, or 2 for legacy checkpoints')


def add_event_arguments(parser):
    for name, default in EVENT_DEFAULTS.items():
        parser.add_argument('--' + name.replace('_', '-'), type=int,
                            help=f'Actual optimizer updates; saved setting or {default} by default')


def event_cadence(raw, overrides=None):
    # Legacy eval/comparison intervals used single-host reference steps. Do not
    # interpret those values as actual distributed optimizer-update intervals.
    saved = raw.get('jaccl_events', {})
    if not isinstance(saved, dict) or set(saved) - EVENT_DEFAULTS.keys():
        raise ValueError('Invalid saved JACCL event cadence')
    result = dict(EVENT_DEFAULTS, **saved)
    for name, value in (overrides or {}).items():
        if name in result and value is not None:
            result[name] = value
    for name, value in result.items():
        minimum = 1 if name == 'eval_every_updates' else 0
        if type(value) is not int or value < minimum:
            raise ValueError(f'{name} must be an integer >= {minimum}')
    return result


def event_cli_args(cadence):
    return [item for name in EVENT_DEFAULTS
            for item in ('--' + name.replace('_', '-'), str(cadence[name]))]


def worker_failed(line):
    match = re.search(r'Node with rank \d+ exited with code (-?\d+)', line)
    return bool((match and int(match.group(1)) != 0) or re.search(r'Node with rank \d+ was killed', line))


def next_boundary(tokens, interval):
    if interval <= 0:
        return None
    return (tokens // interval + 1) * interval


def continuation(metadata, batches=None, grad_accum=None):
    raw = metadata['args']
    saved = metadata.get('distributed_training')
    if saved:
        previous_batches = batch_layout(saved['batches'])
        previous_accum = accumulation_steps(saved.get('grad_accum', 2))
        if (saved['format'] != FORMAT or saved['data_mode'] != 'rank0_contiguous'
                or saved['data_world'] != 1 or metadata['world'] != 4
                or saved['tokens_per_update'] != sum(previous_batches) * previous_accum * 256):
            raise ValueError('Unsupported distributed checkpoint configuration')
        reference = saved['reference_tokens_per_update']
        budget = saved['planned_tokens']
    else:
        if metadata['world'] != 1:
            raise ValueError('Migration requires a single-host checkpoint or this distributed format')
        reference = raw['batch_size'] * raw['grad_accum'] * raw['max_seq_len']
        budget = reference * raw['max_steps']
        previous_batches = BATCHES
        previous_accum = 2
    # args retains the original single-host schedule. Actual distributed batch
    # and accumulation live in distributed_training, never rewriting history.
    fresh_schedule = saved is not None and saved.get('schedule_units') == 'optimizer_updates'
    if fresh_schedule:
        original_budget = reference * raw['max_steps']
        extension = metadata.get('token_extension')
        if extension:
            requested = extension.get('requested_tokens')
            previous = extension.get('previous_planned_tokens')
            if (type(requested) is not int or type(previous) is not int
                    or type(reference) is not int or reference <= 0
                    or extension.get('format') != 'sml-token-extension-v1'
                    or extension.get('original_planned_tokens') != original_budget
                    or not original_budget <= previous < requested
                    or previous % reference != 0
                    or budget != ((requested + reference - 1) // reference) * reference
                    or extension.get('planned_tokens') != budget):
                raise ValueError('Invalid token extension budget')
        if (type(reference) is not int or reference <= 0 or raw['max_seq_len'] != 256
                or reference != raw['batch_size'] * raw['grad_accum'] * 256
                or (not extension and budget != original_budget)):
            raise ValueError('Invalid fresh-run token schedule')
    elif reference != 16384 or raw['grad_accum'] != 2 or raw['max_seq_len'] != 256:
        raise ValueError('Expected the original 16,384-token schedule and sequence 256')
    tokens = metadata['tokens_processed']
    if not isinstance(tokens, int) or tokens < 0 or tokens > budget:
        raise ValueError('Invalid cumulative token count')
    selected_batches = batch_layout(previous_batches if batches is None else batches)
    selected_accum = accumulation_steps(previous_accum if grad_accum is None else grad_accum)
    result = dict(format=FORMAT, batches=selected_batches, grad_accum=selected_accum,
                data_mode='rank0_contiguous',
                data_world=1, reference_tokens_per_update=reference, planned_tokens=budget,
                tokens_per_update=sum(selected_batches) * selected_accum * 256,
                migration_step=saved['migration_step'] if saved else metadata['step'],
                migration_tokens=saved['migration_tokens'] if saved else tokens)
    if fresh_schedule:
        result['schedule_units'] = 'optimizer_updates'
    return result


def validate_cursor(state, metadata):
    if metadata.get('web_phase') and metadata.get('prose_phase'):
        raise ValueError('Ambiguous continuation phase')
    phase = metadata.get('web_phase') or metadata.get('prose_phase')
    offset = phase['start_tokens'] if phase else 0
    if phase:
        labels = {'sml-dolma-web-continuation-v1': ['web'],
                  'sml-dolma-prose-continuation-v1': ['web', 'wiki', 'news', 'books']}.get(phase.get('format'))
        if labels is None or state['source_labels'] != labels or offset < 0:
            raise ValueError('Invalid continuation cursor identity')
        if len(state['source_tokens_emitted']) != len(labels) or any(n < 0 for n in state['source_tokens_emitted']):
            raise ValueError('Invalid per-source token counters')
    if state['total_tokens_emitted'] < 0 or state['total_tokens_emitted'] + offset != metadata['tokens_processed']:
        raise ValueError('Consumed data token count differs from model checkpoint')
    if sum(state['source_tokens_emitted']) != state['total_tokens_emitted']:
        raise ValueError('Per-source token counts do not sum to the consumed count')


def rank_bounds(rank, batches=None):
    if rank not in range(4):
        raise ValueError('Expected rank 0..3')
    batches = batch_layout(BATCHES if batches is None else batches)
    return sum(batches[:rank]), sum(batches[:rank + 1])
