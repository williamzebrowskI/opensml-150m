"""Conservative prose selection and exact-resume streaming for Stage B.

These deterministic rules are a topic/quality screen, not a semantic guarantee.
The separate live-data audit is required before the production launcher runs.
"""
from collections import Counter
import copy
import re
from urllib.parse import urlsplit

from .common import fingerprint
from .corpus import accepted, content_key, partition
from .stream import StreamingTokens

POLICY = 'sml-v2-stage-b-prose-filter-v1'
CODE = re.compile(r'```|\b(?:python|javascript|typescript|java|c\+\+|html|css|sql|bash|powershell|'
                  r'programming|programmers?|source code|pseudocode|coding|computer code|'
                  r'algorithms?|software development|api|apis|github|debugging|compiler|'
                  r'command.line|web.scraping|regular expressions?|opencv|tensorflow|pytorch)\b|'
                  r'\b(?:def|class|import)\s+\w+\s*[:(]|=>|</?\w+>', re.I)
MATH = re.compile(r'\b(?:math(?:s|ematics|ematical)?|arithmetic|algebra|geometry|geometric|'
                  r'calculus|trigonometry|equations?|theorems?|polynomials?|logarithms?|'
                  r'probability|probabilities|statistical|statistics|topology|numerical|'
                  r'fractions?|matrices|multiplication|division problems?|word problems?|'
                  r'calculate|calculating|calculations?|compute|computing|derivatives?|'
                  r'integrals?|percentages?|quadratic|proofs?|square root)\b|'
                  r'\\(?:frac|sum|int|begin|sqrt|alpha|beta)\b|'
                  r'\d\s*[+*/=×÷]\s*\d|\b[a-z]\s*=\s*[-\d]', re.I)
BOILERPLATE = re.compile(r'\b(?:as an ai|as a language model|here is the rewritten|'
                         r'output only|subscribe to|accept all cookies|access denied|'
                         r'page not found|enable javascript|jump to navigation|'
                         r'join to answer|full answer|already exists as an alternate|'
                         r'keep in touch with|clients are delighted)\b|^\s*To create\s*:|\bQ:\s*Q:', re.I)


def document_keys(row, text):
    """Exact text plus original IDs/URLs, shared by originals and rewrites."""
    result = {'content:' + content_key(text)}
    nested = row.get('metadata')
    nested = nested if isinstance(nested, dict) else {}
    for name in ('id', 'url'):
        value = row.get(name) or nested.get(name)
        if not isinstance(value, str) or not value.strip():
            continue
        value = value.strip()
        if name == 'url':
            parts = urlsplit(value)
            if not parts.netloc:
                continue
            # Query strings can identify distinct articles; do not discard them.
            value = parts.netloc.lower().removeprefix('www.') + parts.path.rstrip('/')
            if parts.query:
                value += '?' + parts.query
        result.add(name + ':' + content_key(value))
    return result


def select_document(row, source, config):
    """Return (text, provenance keys, rejection reason), without tokenization."""
    generated = source.get('text_field') == 'rollout_results[0].text'
    original = row.get('text')
    if generated:
        if not isinstance(original, str):
            raise ValueError('FinePhrase is missing its original text')
        outputs = row.get('rollout_results')
        if not isinstance(outputs, list) or len(outputs) != 1 or not isinstance(outputs[0], dict):
            raise ValueError('Unexpected FinePhrase rollout schema')
        output = outputs[0]
        text = output.get('text')
        if not isinstance(text, str):
            raise ValueError('FinePhrase generated text is missing')
        if output.get('finish_reason') != 'stop':
            return None, set(), 'incomplete_generation'
        if (len(text) < config['min_chars'] or len(text.encode('utf-8')) > config['max_bytes']):
            return None, set(), 'length'
    else:
        text = accepted(row, source, config)
        if text is None:
            return None, set(), 'source_quality_or_length'
    keys = document_keys(row, original if generated else text)
    keys.add('content:' + content_key(text))
    # Reserve both original and rewritten text. Rewriting cannot move a held-out
    # original into the training partition.
    if any(partition(k.split(':', 1)[1]) != 'train' for k in keys if k.startswith('content:')):
        return None, keys, 'reserved_partition'
    check = '\n'.join(x for x in (original, text, row.get('prompt'), row.get('title')) if isinstance(x, str))
    if CODE.search(check):
        return None, keys, 'code_topic'
    if MATH.search(check) or row.get('has_math') is True:
        return None, keys, 'math_topic'
    if '\x00' in text or '\ufffd' in text or BOILERPLATE.search(text):
        return None, keys, 'boilerplate_or_encoding'
    words = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text)
    if len(words) < (60 if generated else 35):
        return None, keys, 'too_little_prose'
    alpha = [c for c in text if c.isalpha()]
    if not alpha or sum(c.isascii() for c in alpha) / len(alpha) < .95:
        return None, keys, 'non_english_script'
    grams = [tuple(w.lower() for w in words[i:i+8]) for i in range(len(words)-7)]
    if grams and 1 - len(set(grams)) / len(grams) > .20:
        return None, keys, 'repetition'
    if generated or source.get('format_contains'):
        ending = text.rstrip().rstrip('"\'”’)*_]').rstrip()
        if not ending.endswith(('.', '!', '?')):
            return None, keys, 'incomplete_ending'
    if generated and source.get('config') == 'faq':
        # Question-only / dangling-final-question outputs are common failures.
        tail = text.rsplit('?', 1)
        if len(tail) != 2 or len(re.findall(r'\b\w+\b', tail[1])) < 15:
            return None, keys, 'unanswered_faq'
    if generated and source.get('config') == 'tutorial':
        if len(re.findall(r'^\s*(?:[-*•]|\d+[.)]|(?:\*\*)?Step\s+\d+)', text, re.M | re.I)) < 2:
            return None, keys, 'unstructured_tutorial'
    return text, keys, None


class StageBStreamingTokens(StreamingTokens):
    """New token-mixture counters; preserved raw source positions and global count."""
    def __init__(self, manifest, tokenizer, split, seq_len, state, *, exclusions):
        if split != 'train' or state is None:
            raise ValueError('Stage B requires an explicit migrated/resumed training cursor')
        super().__init__(manifest, tokenizer, split, seq_len, state)
        saved = state.get('stage_b')
        if not isinstance(saved, dict) or saved.get('policy') != POLICY:
            raise ValueError('Missing Stage B selection state')
        self.stage = copy.deepcopy(saved)
        self.exclusions = set(exclusions)
        exclusion_hash = fingerprint(sorted(self.exclusions))
        if self.stage['exclusions'] not in (None, exclusion_hash):
            raise ValueError('Rebuilt reference exclusions differ from saved Stage B state')
        self.stage['exclusions'] = exclusion_hash
        self.stage_offsets = self.stage['offsets']
        if (set(self.stage_offsets) != set(self.weights)
                or any(type(v) is not int or v < 0 or v % seq_len for v in self.stage_offsets.values())
                or any(self.offsets[k] != self.stage['parent_offsets'][k] + self.stage_offsets[k]
                       for k in self.weights)):
            raise ValueError('Stage B mixture/global token accounting differs')
        self.provenance_seen = dict.fromkeys(self.stage.pop('seen_keys'))
        self.counts = {k: Counter(v) for k, v in self.stage.pop('selection_counts').items()}
        if set(self.counts) != set(self.sources):
            raise ValueError('Stage B source counters differ')

    def state_dict(self):
        state = super().state_dict()
        state['stage_b'] = dict(copy.deepcopy(self.stage), seen_keys=list(self.provenance_seen),
                               selection_counts={k: dict(v) for k, v in self.counts.items()})
        return state

    def _accepted_document(self, source):
        cursor, counts = self.sources[source], self.counts[source]
        while True:
            if cursor.rows >= self.config['max_rows_per_source']:
                raise RuntimeError(f'{source}: Stage B scan limit reached; no source substitution')
            row = cursor.next()
            counts['scanned'] += 1
            text, keys, reason = select_document(row, cursor.source, self.config)
            if reason:
                counts[reason] += 1
                continue
            if keys & self.exclusions:
                counts['reference_overlap'] += 1
                continue
            if keys & self.provenance_seen.keys():
                counts['duplicate'] += 1
                continue
            # Bounded pilot: explicit limit rather than evicting keys and silently
            # permitting repeated documents late in the run.
            if len(self.provenance_seen) + len(keys) > 3_000_000:
                raise RuntimeError('Stage B provenance memory budget reached; save and review')
            self.provenance_seen.update(dict.fromkeys(sorted(keys)))
            counts['accepted'] += 1
            return text

    def row(self, source=None):
        source = source or min(self.weights, key=lambda s: self.stage_offsets[s] / self.weights[s])
        result = super().row(source)
        self.stage_offsets[source] += self.seq_len
        return result
