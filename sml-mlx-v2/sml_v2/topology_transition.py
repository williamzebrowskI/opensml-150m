"""Explicit four-to-five-rank migration, preserving each global update and state."""
import copy

from .common import fingerprint

FORMAT = 'sml-v2-five-rank-transition-v1'
PHYSICAL_ORDER = [0,1,2,4,3]


def validate_transition(recipe):
    t = recipe['topology_transition']
    keys = {'format','physical_order','parent_step','start_tokens','parent_contract',
            'parent_replica_hash','parent_batches','tokens_per_update'}
    if not isinstance(t,dict) or set(t) != keys or t['format'] != FORMAT:
        raise ValueError('Invalid five-rank transition')
    if t['physical_order'] != PHYSICAL_ORDER or len(recipe['batches']) != 5:
        raise ValueError('Five-rank transition requires the verified physical order')
    for key in ('parent_step','start_tokens','tokens_per_update'):
        if type(t[key]) is not int or t[key] <= 0:
            raise ValueError('Invalid topology transition counter')
    previous = t['parent_batches']
    if (len(previous) != 4 or any(type(b) is not int or b <= 0 for b in previous)
            or sum(previous) != sum(recipe['batches'])):
        raise ValueError('Topology transition must preserve the global microbatch size')
    expected = sum(recipe['batches']) * recipe['grad_accum'] * recipe['model']['max_seq_len']
    if t['tokens_per_update'] != expected or t['start_tokens'] != t['parent_step'] * expected:
        raise ValueError('Topology transition token/update accounting differs')
    if not isinstance(t['parent_contract'],dict) or set(t['parent_contract']) != {'code','recipe','data','tokenizer'}:
        raise ValueError('Missing topology parent contract')


def migrate_recipe(metadata, batches, physical_order=PHYSICAL_ORDER):
    from .recipe import tokens_per_update, validate
    previous = validate(copy.deepcopy(metadata['recipe']))
    if fingerprint(previous) != metadata['v2_contract']['recipe']:
        raise ValueError('Source recipe fingerprint mismatch')
    if (not isinstance(batches,list) or len(batches) != 5
            or any(type(b) is not int or b < 1 for b in batches)
            or physical_order != PHYSICAL_ORDER):
        raise ValueError('Explicit five-rank batches and verified physical order required')
    if 'topology_transition' in previous:
        if batches != previous['batches'] or physical_order != previous['topology_transition']['physical_order']:
            raise ValueError('Cannot change the saved five-rank geometry on resume')
        return previous
    if len(previous['batches']) != 4 or sum(batches) != sum(previous['batches']):
        raise ValueError('Migration must preserve four-rank global update size')
    if not 0 < metadata['tokens'] < previous['target_tokens']:
        raise ValueError('Migration requires an unfinished, trained source checkpoint')
    recipe = copy.deepcopy(previous)
    recipe['batches'] = batches.copy()
    recipe['topology_transition'] = dict(format=FORMAT, physical_order=physical_order.copy(),
        parent_step=metadata['step'], start_tokens=metadata['tokens'],
        parent_contract=copy.deepcopy(metadata['v2_contract']), parent_replica_hash=metadata['replica_hash'],
        parent_batches=previous['batches'].copy(), tokens_per_update=tokens_per_update(previous))
    return validate(recipe)
