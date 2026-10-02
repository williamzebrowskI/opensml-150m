"""Bounded, paired LR experiment preserving the evaluated parent's training state."""
import copy
import math

from .common import fingerprint

KEY = 'lr_trial_transition'
FORMAT = 'sml-v2-lr-trial-v1'


def validate_transition(recipe):
    from .recipe import learning_rate, tokens_per_update, validate
    from .lr_scheduler import PlateauScheduler
    t = recipe[KEY]
    keys = {'format', 'arm', 'parent_step', 'start_tokens', 'parent_contract',
            'parent_replica_hash', 'parent_scheduler', 'validation_fingerprint',
            'baseline_loss', 'start_lr', 'target_lr', 'warmup_steps', 'trial_steps'}
    if not isinstance(t, dict) or set(t) != keys or t['format'] != FORMAT:
        raise ValueError('Invalid LR trial transition')
    parent = copy.deepcopy(recipe)
    parent.pop(KEY)
    validate(parent)
    if 'data_review_transition' not in parent:
        raise ValueError('LR trial requires the completed data-review setup')
    if (not isinstance(t['parent_contract'], dict)
            or set(t['parent_contract']) != {'code', 'recipe', 'data', 'tokenizer'}
            or fingerprint(parent) != t['parent_contract']['recipe']
            or not t['parent_replica_hash'] or not t['validation_fingerprint']):
        raise ValueError('LR trial changed unrelated settings or lost parent provenance')
    for key, minimum in [('parent_step', 1), ('start_tokens', 1), ('warmup_steps', 0), ('trial_steps', 1)]:
        if type(t[key]) is not int or t[key] < minimum:
            raise ValueError('Invalid LR trial counter: ' + key)
    for key in ['baseline_loss', 'start_lr', 'target_lr']:
        if type(t[key]) not in (int, float) or not math.isfinite(t[key]) or t[key] <= 0:
            raise ValueError('Invalid LR trial value: ' + key)
    end_tokens = t['start_tokens'] + t['trial_steps'] * tokens_per_update(parent)
    if end_tokens >= parent['target_tokens'] * parent['decay_start_fraction']:
        raise ValueError('LR trial must finish before the original final decay')
    state = PlateauScheduler(parent['plateau_scheduler'], best=t['baseline_loss'],
                             step=t['parent_step'], tokens=t['start_tokens'],
                             state=t['parent_scheduler']).state_dict()
    if (state['last_step'] != t['parent_step'] or state['last_tokens'] != t['start_tokens']
            or state['last_loss'] != t['baseline_loss']):
        raise ValueError('LR trial requires an evaluated best checkpoint')
    if t['start_lr'] != min(learning_rate(t['start_tokens'], parent), state['cap']):
        raise ValueError('LR trial starting rate differs from the parent')
    if t['arm'] == 'rewarm':
        if not (0 < t['warmup_steps'] < t['trial_steps']
                and t['start_lr'] < t['target_lr'] <= parent['plateau_scheduler']['initial_lr']
                and t['target_lr'] <= learning_rate(end_tokens, parent)):
            raise ValueError('Invalid LR trial warmup or target')
    elif t['arm'] == 'control':
        if t['warmup_steps'] != 0 or t['target_lr'] != t['start_lr']:
            raise ValueError('Control must preserve the parent learning rate')
    else:
        raise ValueError('Unknown LR trial arm')


def trial_recipe(metadata, arm='rewarm', *, target_lr=1e-5, warmup_steps=100, trial_steps=1000):
    from .recipe import learning_rate, validate
    from .lr_scheduler import restore_scheduler
    previous = validate(copy.deepcopy(metadata['recipe']))
    if metadata.get('sft') or fingerprint(previous) != metadata['v2_contract']['recipe']:
        raise ValueError('LR trial requires a verified pretraining checkpoint')
    if KEY in previous:
        t = previous[KEY]
        expected = (arm, target_lr, warmup_steps, trial_steps) if arm == 'rewarm' else (arm, t['start_lr'], 0, trial_steps)
        if (t['arm'], t['target_lr'], t['warmup_steps'], t['trial_steps']) != expected:
            raise ValueError('Saved LR trial differs; refusing to restart or reconfigure it')
        return previous
    scheduler = restore_scheduler(previous, metadata)
    if scheduler is None:
        raise ValueError('LR trial requires a retained scheduler')
    start_lr = scheduler.rate(learning_rate(metadata['tokens'], previous))
    previous[KEY] = dict(format=FORMAT, arm=arm, parent_step=metadata['step'],
        start_tokens=metadata['tokens'], parent_contract=copy.deepcopy(metadata['v2_contract']),
        parent_replica_hash=metadata['replica_hash'], parent_scheduler=scheduler.state_dict(),
        validation_fingerprint=metadata['validation_fingerprint'], baseline_loss=metadata['best_val_loss'],
        start_lr=start_lr, target_lr=target_lr if arm == 'rewarm' else start_lr,
        warmup_steps=warmup_steps if arm == 'rewarm' else 0, trial_steps=trial_steps)
    return validate(previous)


def end_step(recipe):
    t = recipe.get(KEY)
    return t['parent_step'] + t['trial_steps'] if t else None


def apply_ramp(tokens, recipe, base_lr):
    from .recipe import tokens_per_update
    t = recipe.get(KEY)
    if not t:
        return base_lr
    span = t['warmup_steps'] * tokens_per_update(recipe)
    progress = min(1., max(0., (tokens - t['start_tokens']) / span)) if span else 1.
    return min(base_lr, t['start_lr'] + (t['target_lr'] - t['start_lr']) * progress)
