"""Explicit Stage B data transition, anchored to an evaluated base checkpoint."""
import copy
import math

from .common import fingerprint
from .stage_b_data import POLICY
from .stream import empty_shuffle_state, stream_manifest

KEY = 'stage_b_transition'
FORMAT = 'sml-v2-stage-b-v1'


def training_manifest(corpus, tokenizer, settings):
    reference = stream_manifest(corpus['reference_corpus'], tokenizer, settings)
    if corpus.get('stage_b_selection') != POLICY:
        raise ValueError('Unknown Stage B document selection policy')
    if settings.get('shuffle_buffer', 0) < 2:
        raise ValueError('Stage B requires resumable document shuffling')
    return dict(format=FORMAT, corpus=copy.deepcopy(corpus), reference=reference,
                tokenizer=tokenizer.fingerprint, settings=copy.deepcopy(settings),
                weights={s['label']: s['weight'] for s in corpus['sources']},
                datasets=reference['datasets'], document_order='bounded-accepted-document-shuffle-v1')


def validate_transition(recipe):
    from .recipe import tokens_per_update, validate
    from .lr_scheduler import PlateauScheduler
    t = recipe[KEY]
    keys = {'format', 'parent_step', 'start_tokens', 'parent_contract', 'parent_replica_hash',
            'parent_scheduler', 'validation_fingerprint', 'baseline_loss', 'manifest',
            'token_budget', 'trial_steps', 'learning_rate'}
    if not isinstance(t, dict) or set(t) != keys or t['format'] != FORMAT:
        raise ValueError('Invalid Stage B transition')
    parent = copy.deepcopy(recipe); parent.pop(KEY)
    validate(parent)
    if 'lr_trial_transition' in parent or 'data_review_transition' not in parent:
        raise ValueError('Stage B must start from the evaluated data-review base, not an LR trial')
    if (fingerprint(parent) != t['parent_contract']['recipe'] or not t['parent_replica_hash']
            or not t['manifest'] or not t['validation_fingerprint']):
        raise ValueError('Stage B parent recipe/provenance changed')
    for key in ('parent_step', 'start_tokens', 'token_budget', 'trial_steps'):
        if type(t[key]) is not int or t[key] < 1:
            raise ValueError('Invalid Stage B counter: ' + key)
    if (t['trial_steps'] != math.ceil(t['token_budget'] / tokens_per_update(parent))
            or t['start_tokens'] + t['trial_steps'] * tokens_per_update(parent) >=
            parent['target_tokens'] * parent['decay_start_fraction']):
        raise ValueError('Stage B budget must end before the original final decay')
    state = PlateauScheduler(parent['plateau_scheduler'], best=t['baseline_loss'],
                             step=t['parent_step'], tokens=t['start_tokens'],
                             state=t['parent_scheduler']).state_dict()
    if (state['last_step'], state['last_tokens'], state['last_loss']) != (
            t['parent_step'], t['start_tokens'], t['baseline_loss']):
        raise ValueError('Stage B requires an evaluated best checkpoint')
    if (not math.isfinite(t['learning_rate']) or t['learning_rate'] != state['cap']
            or t['learning_rate'] != parent['plateau_scheduler']['min_lr']):
        raise ValueError('This data-only Stage B pilot retains the settled parent LR floor')


def stage_recipe(metadata, manifest, token_budget=250_000_000):
    from .recipe import validate, tokens_per_update
    previous = validate(copy.deepcopy(metadata['recipe']))
    if metadata.get('sft') or fingerprint(previous) != metadata['v2_contract']['recipe']:
        raise ValueError('Stage B requires a verified pretraining source')
    if KEY in previous:
        t = previous[KEY]
        if t['manifest'] != fingerprint(manifest) or t['token_budget'] != token_budget:
            raise ValueError('Saved Stage B mixture/budget differs; refusing a restart or reconfiguration')
        return previous
    if fingerprint(manifest['reference']) != metadata['v2_contract']['data']:
        raise ValueError('Stage B reference corpus differs from the parent data')
    previous[KEY] = dict(format=FORMAT, parent_step=metadata['step'], start_tokens=metadata['tokens'],
        parent_contract=copy.deepcopy(metadata['v2_contract']), parent_replica_hash=metadata['replica_hash'],
        parent_scheduler=copy.deepcopy(metadata['lr_scheduler']),
        validation_fingerprint=metadata['validation_fingerprint'], baseline_loss=metadata['best_val_loss'],
        manifest=fingerprint(manifest), token_budget=token_budget,
        trial_steps=math.ceil(token_budget / tokens_per_update(previous)), learning_rate=metadata['lr_scheduler']['cap'])
    return validate(previous)


def verify_manifest(metadata, manifest, contract, recipe):
    if (manifest is None or manifest.get('format') != FORMAT
            or fingerprint(manifest) != contract['data']
            or manifest['tokenizer'] != contract['tokenizer']
            or manifest['settings'] != recipe['stream']
            or fingerprint(manifest['reference']) != metadata['v2_contract']['data']):
        raise ValueError('Stage B changed the tokenizer/reference or lacks its actual data manifest')
    if recipe != stage_recipe(metadata, manifest, recipe[KEY]['token_budget']):
        raise ValueError('Stage B changed unrelated settings')


def migrate_cursor(state, metadata, manifest):
    """Keep raw HF positions; explicitly discard pre-filter pending input only.

    Old pending documents/token tails have not undergone the new selection. Do
    not train them silently, replay consumed rows, or rebalance lifetime counts.
    """
    if (state['fingerprint'] != metadata['v2_contract']['data'] or state['split'] != 'train'
            or sum(state['offsets'].values()) != metadata['tokens']
            or fingerprint(manifest['reference']) != metadata['v2_contract']['data']):
        raise ValueError('Stage B source cursor mismatch')
    old_sources = {s['label']: s for s in manifest['reference']['corpus']['sources']}
    for source in manifest['corpus']['sources']:
        if source['label'] in old_sources:
            expected = dict(old_sources[source['label']], weight=source['weight'])
            if source != expected:
                raise ValueError('An existing Stage B source changed beyond its token weight')
    result = copy.deepcopy(state)
    result['fingerprint'] = fingerprint(manifest)
    dropped = {}
    for label in manifest['weights']:
        if label in result['sources']:
            dropped[label] = dict(token_tail=len(result['sources'][label]['buffer']),
                pending_documents=len(result['shuffle']['sources'][label]['documents']))
            result['sources'][label]['buffer'] = []
        else:
            result['sources'][label] = dict(dataset=None, rows=0, buffer=[])
            result['offsets'][label] = 0
    result['shuffle'] = empty_shuffle_state(manifest)
    result['stage_b'] = dict(policy=POLICY, exclusions=None,
        parent_offsets=copy.deepcopy(result['offsets']), offsets={k: 0 for k in manifest['weights']},
        discarded_unconsumed_input=dropped,
        seen_keys=['content:' + h for h in result['recent_document_hashes']],
        selection_counts={k: {} for k in manifest['weights']})
    return result


def end_step(recipe):
    t = recipe.get(KEY)
    return t['parent_step'] + t['trial_steps'] if t else None
