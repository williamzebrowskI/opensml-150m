"""Token-based learning schedule and strict experiment settings."""
import math


def validate(recipe):
    if 'stage_b_transition' in recipe:
        from .stage_b import validate_transition
        validate_transition(recipe)
        return recipe
    if 'lr_trial_transition' in recipe:
        # Validate historical transitions against their original recipe, then
        # validate this additional layer's exact parent fingerprint.
        from .lr_trial import validate_transition
        validate_transition(recipe)
        return recipe
    mode = recipe.get('data_mode', 'tokens')
    if mode not in ('tokens', 'hf_stream'):
        raise ValueError('Unknown data mode')
    if mode == 'hf_stream':
        for key, minimum in (('prefetch_batches', 0), ('read_attempts', 1), ('dedup_window', 1)):
            value = recipe.get('stream', {}).get(key)
            if type(value) is not int or value < minimum:
                raise ValueError(f'Invalid streaming setting: {key}')
        capacity = recipe['stream'].get('shuffle_buffer', 0)
        if type(capacity) is not int or capacity < 0 or capacity > 10000 or capacity == 1:
            raise ValueError('Invalid streaming shuffle buffer')
    for key in ('target_tokens', 'grad_accum', 'log_every', 'eval_every_tokens',
                'sample_every_tokens', 'save_every_tokens', 'eval_batches_per_source',
                'eval_batch_size', 'keep_checkpoints'):
        if type(recipe[key]) is not int or recipe[key] < 1:
            raise ValueError(f'Invalid positive integer: {key}')
    if len(recipe['batches']) not in (1, 4, 5) or any(type(b) is not int or b < 1 for b in recipe['batches']):
        raise ValueError('Use one local rank or four/five positive ring microbatches')
    if len(recipe['batches']) == 4 and len(set(recipe['batches'][1:])) != 1:
        raise ValueError('Physical ring order requires equal microbatches on the three peers')
    if not 0 < recipe['warmup_fraction'] < recipe['decay_start_fraction'] < 1:
        raise ValueError('Require 0 < warmup < decay_start < 1')
    if not 0 < recipe['min_lr_ratio'] <= 1 or not 0 < recipe['peak_lr'] < 1:
        raise ValueError('Invalid learning rate settings')
    if not 0 < recipe['grad_clip'] or not 0 <= recipe['reserve_gib']:
        raise ValueError('Invalid clipping/disk reserve')
    if 'continuation' in recipe:
        extension = recipe['continuation']
        if extension.get('format') != 'sml-v2-continuation-v1':
            raise ValueError('Unknown continuation format')
        for key, minimum in (('start_tokens', 0), ('parent_step', 0), ('rewarm_tokens', 1)):
            if type(extension.get(key)) is not int or extension[key] < minimum:
                raise ValueError(f'Invalid continuation setting: {key}')
        lr = extension.get('start_lr')
        if not isinstance(lr, (int, float)) or not math.isfinite(lr) or not 0 <= lr <= recipe['peak_lr']:
            raise ValueError('Invalid continuation starting learning rate')
        if extension['start_tokens'] + extension['rewarm_tokens'] >= recipe['target_tokens'] * recipe['decay_start_fraction']:
            raise ValueError('Continuation ramp must finish before final decay begins')
    if 'plateau_scheduler' in recipe:
        from .lr_scheduler import TRANSITION, validate_config
        validate_config(recipe['plateau_scheduler'], recipe['peak_lr'])
        transition = recipe.get('scheduler_transition', {})
        if transition.get('format') != TRANSITION:
            raise ValueError('Plateau scheduler requires explicit transition provenance')
        for key in ('parent_step', 'start_tokens'):
            if type(transition.get(key)) is not int or transition[key] < 1:
                raise ValueError(f'Invalid scheduler transition: {key}')
        if transition['start_tokens'] >= recipe['target_tokens']:
            raise ValueError('Scheduler transition must precede the token target')
    elif 'scheduler_transition' in recipe:
        raise ValueError('Scheduler transition without a scheduler')
    if 'topology_transition' in recipe:
        from .topology_transition import validate_transition
        validate_transition(recipe)
    if 'lr_floor_transition' in recipe:
        if 'plateau_scheduler' not in recipe:
            raise ValueError('LR floor transition without a scheduler')
        from .lr_scheduler import validate_floor_transition
        validate_floor_transition(recipe)
    if 'data_review_transition' in recipe:
        from .data_review import validate_transition
        validate_transition(recipe)
    return recipe


def tokens_per_update(recipe):
    return sum(recipe['batches']) * recipe['grad_accum'] * recipe['model']['max_seq_len']


def learning_rate(tokens, recipe):
    if 'stage_b_transition' in recipe:
        return recipe['stage_b_transition']['learning_rate']
    from .lr_trial import apply_ramp
    return apply_ramp(tokens, recipe, _token_learning_rate(tokens, recipe))


def _token_learning_rate(tokens, recipe):
    if 'continuation' in recipe:
        extension = recipe['continuation']
        ramp_end = extension['start_tokens'] + extension['rewarm_tokens']
        if tokens < ramp_end:
            progress = max(0, (tokens - extension['start_tokens']) / extension['rewarm_tokens'])
            return extension['start_lr'] + (recipe['peak_lr'] - extension['start_lr']) * progress
        decay_start = recipe['target_tokens'] * recipe['decay_start_fraction']
        if tokens < decay_start:
            return recipe['peak_lr']
        progress = min(1, (tokens - decay_start) / (recipe['target_tokens'] - decay_start))
        return recipe['peak_lr'] * (recipe['min_lr_ratio'] +
            (1 - recipe['min_lr_ratio']) * (1 + math.cos(math.pi * progress)) / 2)
    progress = min(1, max(0, tokens / recipe['target_tokens']))
    warm, decay = recipe['warmup_fraction'], recipe['decay_start_fraction']
    if progress < warm:
        scale = progress / warm
    elif progress < decay:
        scale = 1.0
    else:
        scale = recipe['min_lr_ratio'] + (1 - recipe['min_lr_ratio']) * (1 + math.cos(math.pi * (progress-decay)/(1-decay))) / 2
    return recipe['peak_lr'] * scale


def event_due(before, after, interval):
    return before // interval != after // interval
