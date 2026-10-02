"""Explicit, cursor-preserving document shuffle and validation expansion."""
import copy
import math

from .common import fingerprint
from .stream import empty_shuffle_state, with_stream_settings

FORMAT = 'sml-v2-data-review-v1'


def validate_transition(recipe):
    t = recipe['data_review_transition']
    keys = {'format', 'parent_step', 'start_tokens', 'parent_contract', 'parent_replica_hash',
            'parent_stream', 'legacy_eval_batches', 'legacy_validation_fingerprint',
            'legacy_best_val_loss', 'parent_scheduler'}
    if not isinstance(t, dict) or set(t) != keys or t['format'] != FORMAT:
        raise ValueError('Invalid data review transition')
    if recipe.get('data_mode') != 'hf_stream' or 'plateau_scheduler' not in recipe:
        raise ValueError('Data review requires streaming and a retained scheduler')
    if len(recipe['batches']) not in (1, 5):
        raise ValueError('Data review requires an existing local or five-rank run')
    for key in ('parent_step', 'start_tokens', 'legacy_eval_batches'):
        if type(t[key]) is not int or t[key] < 1:
            raise ValueError('Invalid data review counter')
    if t['start_tokens'] >= recipe['target_tokens']:
        raise ValueError('Data review requires an unfinished run')
    if (not isinstance(t['parent_contract'], dict)
            or set(t['parent_contract']) != {'code', 'recipe', 'data', 'tokenizer'}
            or not t['parent_replica_hash'] or not t['legacy_validation_fingerprint']):
        raise ValueError('Missing data review provenance')
    if (not isinstance(t['parent_stream'], dict) or t['parent_stream'].get('shuffle_buffer', 0)
            or recipe['stream'] != dict(t['parent_stream'], shuffle_buffer=recipe['stream'].get('shuffle_buffer'))
            or not 2 <= recipe['stream'].get('shuffle_buffer', 0) <= 10000
            or recipe['eval_batches_per_source'] <= t['legacy_eval_batches']):
        raise ValueError('Data review must only add shuffling and enlarge validation')
    best = t['legacy_best_val_loss']
    if type(best) not in (int, float) or not math.isfinite(best) or best < 0:
        raise ValueError('Missing finite legacy validation best')
    parent = copy.deepcopy(recipe)
    parent.pop('data_review_transition')
    parent['stream'] = t['parent_stream']
    parent['eval_batches_per_source'] = t['legacy_eval_batches']
    if fingerprint(parent) != t['parent_contract']['recipe']:
        raise ValueError('Data review changed unrelated recipe settings')
    from .lr_scheduler import PlateauScheduler
    PlateauScheduler(recipe['plateau_scheduler'], best=best, step=t['parent_step'],
                     tokens=t['start_tokens'], state=t['parent_scheduler'])


def reviewed_recipe(metadata, shuffle_buffer=256, eval_batches=256):
    from .recipe import validate
    from .lr_scheduler import restore_scheduler
    previous = validate(copy.deepcopy(metadata['recipe']))
    if metadata.get('sft') or fingerprint(previous) != metadata['v2_contract']['recipe']:
        raise ValueError('Data review requires a verified pretraining recipe')
    if 'data_review_transition' in previous:
        if (previous['stream']['shuffle_buffer'], previous['eval_batches_per_source']) != (shuffle_buffer, eval_batches):
            raise ValueError('Saved data review settings differ; refusing a second transition')
        return previous
    scheduler = restore_scheduler(previous, metadata)
    if scheduler is None:
        raise ValueError('Data review requires a retained scheduler')
    recipe = copy.deepcopy(previous)
    recipe['stream'] = dict(previous['stream'], shuffle_buffer=shuffle_buffer)
    recipe['eval_batches_per_source'] = eval_batches
    recipe['data_review_transition'] = dict(
        format=FORMAT, parent_step=metadata['step'], start_tokens=metadata['tokens'],
        parent_contract=copy.deepcopy(metadata['v2_contract']), parent_replica_hash=metadata['replica_hash'],
        parent_stream=copy.deepcopy(previous['stream']), legacy_eval_batches=previous['eval_batches_per_source'],
        legacy_validation_fingerprint=metadata['validation_fingerprint'],
        legacy_best_val_loss=metadata['best_val_loss'], parent_scheduler=scheduler.state_dict())
    return validate(recipe)


def verify_manifest(metadata, manifest, contract):
    if manifest is None or fingerprint(manifest) != contract['data']:
        raise ValueError('Data review requires the actual new streaming manifest')
    settings = manifest['settings']
    if (settings != dict(metadata['recipe']['stream'], shuffle_buffer=settings.get('shuffle_buffer'))
            or manifest != with_stream_settings(manifest, settings)):
        raise ValueError('Data review changed unrelated stream settings or document order')
    parent = with_stream_settings(manifest, metadata['recipe']['stream'])
    if fingerprint(parent) != metadata['v2_contract']['data']:
        raise ValueError('Data review changed corpus, tokenizer, packages or unrelated stream settings')


def migrate_cursor(state, metadata, manifest):
    if (state['fingerprint'] != metadata['v2_contract']['data'] or state['split'] != 'train'
            or state['seq_len'] != metadata['recipe']['model']['max_seq_len']
            or sum(state['offsets'].values()) != metadata['tokens'] or 'shuffle' in state):
        raise ValueError('Data review source cursor mismatch')
    verify_manifest(metadata, manifest, dict(data=fingerprint(manifest)))
    result = copy.deepcopy(state)
    result['fingerprint'] = fingerprint(manifest)
    result['shuffle'] = empty_shuffle_state(manifest)
    return result


def rebaseline_scheduler(scheduler, loss, step, tokens):
    """Change the monitored metric, retaining the LR cap and lifetime counters."""
    from .lr_scheduler import PlateauScheduler
    state = scheduler.state_dict()
    state.update(best=loss, last_loss=loss, last_step=step, last_tokens=tokens,
                 bad_evals=0, cooldown_remaining=0)
    return PlateauScheduler(scheduler.config, best=loss, step=step, tokens=tokens, state=state)
