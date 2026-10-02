"""Checkpointed validation controller; caps, but never raises, the token schedule."""

import copy
import math

from .common import fingerprint

FORMAT = 'sml-v2-plateau-v1'
TRANSITION = 'sml-v2-scheduler-transition-v1'
FLOOR_TRANSITION = 'sml-v2-lr-floor-transition-v1'
DEFAULTS = dict(format=FORMAT, initial_lr=2e-4, factor=0.5, patience=8,
                cooldown=2, min_delta=0.002, min_lr=3e-5)


def validate_config(config, peak_lr):
    if not isinstance(config, dict) or set(config) != set(DEFAULTS) or config.get('format') != FORMAT:
        raise ValueError('Invalid plateau scheduler configuration')
    for key in ('initial_lr', 'factor', 'min_delta', 'min_lr'):
        value = config[key]
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f'Invalid plateau scheduler setting: {key}')
    for key, minimum in (('patience', 1), ('cooldown', 0)):
        if type(config[key]) is not int or config[key] < minimum:
            raise ValueError(f'Invalid plateau scheduler setting: {key}')
    if not (0 < config['min_lr'] <= config['initial_lr'] <= peak_lr):
        raise ValueError('Require 0 < minimum LR <= initial LR <= peak LR')
    if not (0 < config['factor'] < 1) or config['min_delta'] < 0:
        raise ValueError('Invalid plateau reduction factor or improvement threshold')
    return config


def adaptive_recipe(metadata, config):
    """Derive exactly one explicit scheduler change, anchored to its parent bundle."""
    from .recipe import learning_rate, validate
    previous = validate(copy.deepcopy(metadata['recipe']))
    if fingerprint(previous) != metadata['v2_contract']['recipe']:
        raise ValueError('Source checkpoint recipe fingerprint mismatch')
    validate_config(config, previous['peak_lr'])
    if 'plateau_scheduler' in previous:
        if previous['plateau_scheduler'] != config:
            raise ValueError('Saved scheduler differs; refusing to reset or reconfigure it')
        return previous
    if not 0 < metadata['tokens'] < previous['target_tokens']:
        raise ValueError('Scheduler transition requires an unfinished pretrained checkpoint')
    if config['initial_lr'] > learning_rate(metadata['tokens'], previous):
        raise ValueError('Scheduler transition must not raise the current learning rate')
    if config['min_lr'] > previous['peak_lr'] * previous['min_lr_ratio'] + 1e-15:
        raise ValueError('Scheduler floor must not exceed the existing final LR')
    best = metadata.get('best_val_loss')
    if type(best) not in (int, float) or not math.isfinite(best) or best < 0:
        raise ValueError('Scheduler transition requires a finite saved validation best')
    previous['plateau_scheduler'] = copy.deepcopy(config)
    previous['scheduler_transition'] = dict(
        format=TRANSITION, parent_step=metadata['step'], start_tokens=metadata['tokens'],
        parent_contract=copy.deepcopy(metadata['v2_contract']),
        parent_replica_hash=metadata['replica_hash'], baseline_best=best)
    return validate(previous)


def validate_floor_transition(recipe):
    """Validate the recorded, one-time reduction from an exhausted LR floor."""
    t = recipe['lr_floor_transition']
    keys = {'format', 'parent_step', 'start_tokens', 'parent_contract',
            'parent_replica_hash', 'previous_min_lr', 'resume_cap', 'parent_scheduler'}
    if not isinstance(t, dict) or set(t) != keys or t['format'] != FLOOR_TRANSITION:
        raise ValueError('Invalid LR floor transition')
    if len(recipe['batches']) != 5 or 'topology_transition' not in recipe:
        raise ValueError('LR floor transition requires an existing five-rank run')
    for key in ('parent_step', 'start_tokens'):
        if type(t[key]) is not int or t[key] <= 0:
            raise ValueError('Invalid LR floor transition counter')
    if t['start_tokens'] >= recipe['target_tokens']:
        raise ValueError('LR floor transition requires an unfinished run')
    if not isinstance(t['parent_contract'], dict) or set(t['parent_contract']) != {'code', 'recipe', 'data', 'tokenizer'}:
        raise ValueError('Missing LR floor parent contract')
    if not isinstance(t['parent_replica_hash'], str) or not t['parent_replica_hash']:
        raise ValueError('Missing LR floor parent replica hash')
    config = recipe['plateau_scheduler']
    old_config = dict(config, min_lr=t['previous_min_lr'])
    validate_config(old_config, recipe['peak_lr'])
    if not config['min_lr'] < old_config['min_lr']:
        raise ValueError('The new LR floor must be lower')
    if not isinstance(t['parent_scheduler'], dict):
        raise ValueError('Missing parent scheduler history')
    state = PlateauScheduler(old_config, best=0, step=t['parent_step'],
                             tokens=t['start_tokens'], state=t['parent_scheduler']).state_dict()
    if state['cap'] != old_config['min_lr'] or state['cooldown_remaining']:
        raise ValueError('Lower the LR floor only after reaching and settling at the old floor')
    if t['resume_cap'] != max(config['min_lr'], state['cap'] * config['factor']):
        raise ValueError('LR floor transition must apply exactly one standard reduction')


def lower_floor_recipe(metadata, min_lr):
    """Keep all training settings, with an explicit lower floor and one LR cut.

    Repeating the option on a checkpoint from this continuation is idempotent:
    its saved cap and controller history take precedence over the initial cut.
    """
    from .recipe import validate
    if metadata.get('sft'):
        raise ValueError('LR floor continuation requires a pretraining checkpoint')
    previous = validate(copy.deepcopy(metadata['recipe']))
    if fingerprint(previous) != metadata['v2_contract']['recipe']:
        raise ValueError('Source checkpoint recipe fingerprint mismatch')
    if type(min_lr) not in (int, float) or not math.isfinite(min_lr) or min_lr <= 0:
        raise ValueError('LR floor must be finite and positive')
    config = previous.get('plateau_scheduler')
    if config is None:
        raise ValueError('LR floor continuation requires a saved plateau scheduler')
    if 'lr_floor_transition' in previous:
        if min_lr != config['min_lr']:
            raise ValueError('Saved LR floor differs; refusing to reconfigure this continuation')
        return previous
    if min_lr >= config['min_lr']:
        raise ValueError('The new LR floor must be lower')
    scheduler = restore_scheduler(previous, metadata)
    state = scheduler.state_dict()
    previous['plateau_scheduler'] = dict(config, min_lr=min_lr)
    previous['lr_floor_transition'] = dict(
        format=FLOOR_TRANSITION, parent_step=metadata['step'], start_tokens=metadata['tokens'],
        parent_contract=copy.deepcopy(metadata['v2_contract']),
        parent_replica_hash=metadata['replica_hash'], previous_min_lr=config['min_lr'],
        resume_cap=max(min_lr, state['cap'] * config['factor']), parent_scheduler=state)
    return validate(previous)


class PlateauScheduler:
    def __init__(self, config, *, best, step, tokens, state=None):
        self.config = copy.deepcopy(validate_config(config, 1.0))
        if state is None:
            state = dict(format=FORMAT, config_fingerprint=fingerprint(config),
                         cap=config['initial_lr'], best=best, bad_evals=0,
                         cooldown_remaining=config['cooldown'], reductions=0,
                         evaluations=0, last_step=step, last_tokens=tokens, last_loss=None)
        self.state = copy.deepcopy(state)
        self._validate_state(step, tokens)

    def _validate_state(self, step, tokens):
        s, c = self.state, self.config
        keys = {'format', 'config_fingerprint', 'cap', 'best', 'bad_evals',
                'cooldown_remaining', 'reductions', 'evaluations', 'last_step',
                'last_tokens', 'last_loss'}
        if (not isinstance(s, dict) or set(s) != keys or s['format'] != FORMAT or
                s['config_fingerprint'] != fingerprint(c)):
            raise ValueError('Missing or incompatible scheduler state')
        for key in ('cap', 'best'):
            if type(s[key]) not in (int, float) or not math.isfinite(s[key]) or s[key] < 0:
                raise ValueError(f'Invalid scheduler state: {key}')
        for key in ('bad_evals', 'cooldown_remaining', 'reductions', 'evaluations', 'last_step', 'last_tokens'):
            if type(s[key]) is not int or s[key] < 0:
                raise ValueError(f'Invalid scheduler state: {key}')
        if (not c['min_lr'] <= s['cap'] <= c['initial_lr'] or s['bad_evals'] >= c['patience'] or
                s['cooldown_remaining'] > c['cooldown'] or s['last_step'] > step or
                s['last_tokens'] > tokens or s['reductions'] > s['evaluations']):
            raise ValueError('Scheduler state counters or LR are inconsistent')
        loss = s['last_loss']
        if loss is not None and (type(loss) not in (int, float) or not math.isfinite(loss) or loss < 0):
            raise ValueError('Invalid scheduler last validation loss')

    def state_dict(self):
        return copy.deepcopy(self.state)

    def rate(self, base_lr):
        if not math.isfinite(base_lr) or base_lr < 0:
            raise ValueError('Invalid base learning rate')
        return min(base_lr, self.state['cap'])

    def observe(self, loss, *, step, tokens):
        s, c = self.state, self.config
        if type(loss) not in (int, float) or not math.isfinite(loss) or loss < 0:
            raise ValueError('Nonfinite/invalid validation loss; scheduler not advanced')
        if type(step) is not int or type(tokens) is not int or step <= s['last_step'] or tokens <= s['last_tokens']:
            raise ValueError('Duplicate or out-of-order scheduler evaluation')
        s.update(last_loss=loss, last_step=step, last_tokens=tokens, evaluations=s['evaluations'] + 1)
        improved = loss < s['best'] - c['min_delta']
        if improved:
            s['best'], s['bad_evals'] = loss, 0
        if s['cooldown_remaining']:
            s['cooldown_remaining'] -= 1
            s['bad_evals'] = 0
            return 'cooldown'
        if improved:
            return 'improved'
        if s['cap'] <= c['min_lr']:
            s['bad_evals'] = 0
            return 'at-floor'
        s['bad_evals'] += 1
        if s['bad_evals'] < c['patience']:
            return 'monitoring'
        s['cap'] = max(c['min_lr'], s['cap'] * c['factor'])
        s['bad_evals'], s['cooldown_remaining'] = 0, c['cooldown']
        s['reductions'] += 1
        return 'reduced'


def restore_scheduler(recipe, metadata):
    config = recipe.get('plateau_scheduler')
    if config is None:
        if metadata and metadata.get('lr_scheduler') is not None:
            raise ValueError('Cannot silently remove saved scheduler state')
        return None
    if metadata is None:
        raise ValueError('Validation scheduler requires an explicit checkpoint transition')
    if 'plateau_scheduler' in metadata['recipe']:
        if not isinstance(metadata.get('lr_scheduler'), dict):
            raise ValueError('Checkpoint is missing required scheduler state')
        state = copy.deepcopy(metadata['lr_scheduler'])
        if 'lr_trial_transition' in recipe and 'lr_trial_transition' not in metadata['recipe']:
            from .lr_trial import trial_recipe
            t = recipe['lr_trial_transition']
            expected = trial_recipe(metadata, t['arm'], target_lr=t['target_lr'],
                                    warmup_steps=t['warmup_steps'], trial_steps=t['trial_steps'])
            if recipe != expected:
                raise ValueError('LR trial changed unrelated settings or provenance')
            # The recipe applies the absolute-token ramp. Retain all controller
            # history and preserve any later reductions on subsequent resumes.
            state['cap'] = t['target_lr']
        if 'lr_floor_transition' in recipe and 'lr_floor_transition' not in metadata['recipe']:
            expected = lower_floor_recipe(metadata, config['min_lr'])
            if recipe != expected:
                raise ValueError('LR floor transition changed unrelated settings or provenance')
            state.update(config_fingerprint=fingerprint(config),
                         cap=recipe['lr_floor_transition']['resume_cap'], bad_evals=0,
                         cooldown_remaining=config['cooldown'], reductions=state['reductions'] + 1)
    else:
        state = None
    return PlateauScheduler(config, best=metadata['best_val_loss'], step=metadata['step'],
                            tokens=metadata['tokens'], state=state)
