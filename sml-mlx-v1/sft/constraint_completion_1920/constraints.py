"""Conservative parser for a subset of Smol-Constraints, not an IFEval scorer.

Unknown constraint language is rejected. Surface checks never certify factual
correctness or that the requested task was fully answered.
"""
import re
from collections import Counter

FLAGS = re.I


def words(text):
    return re.findall(r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)*", text)


def sentence_count(text):
    text = re.sub(r'<<[^<>]+>>', '', text)
    text = re.sub(r'(?im)^\s*section\s+\d+\s*[:.]?\s*$', '', text)
    text = re.sub(r'(?i)\bp\.s\.', 'PS', text)
    return len([s for s in re.split(r'[.!?]+(?:["\x27*]*)\s+|[.!?]+(?:["\x27*]*)$', text.strip()) if words(s)])


def compare(actual, mode, n):
    return {'at least': actual >= n, 'exactly': actual == n, 'less than': actual < n,
            'at most': actual <= n, 'more than': actual > n}[mode]


def parse(prompt):
    rules = []
    text = prompt.strip()

    def remove(pattern, convert=None):
        nonlocal text
        def found(m):
            if convert:
                rule = convert(m)
                if rule is not None: rules.append(rule)
            return ' '
        text = re.sub(pattern, found, text, flags=FLAGS)

    remove(r'(?:your |the |this )?(?:response|answer) (?:should|must) (?:also )?contain (at least|exactly|less than|at most|more than) (\d+) (sentences|words|bullet points)\.?',
           lambda m: dict(kind={'sentences':'sentences','words':'words','bullet points':'bullets'}[m[3].lower()],mode=m[1].lower(),n=int(m[2])))
    remove(r'(?:your |the )?(?:response|answer) must have (\d+) sections\.?', lambda m: dict(kind='sections',mode='exactly',n=int(m[1])))
    remove(r'Mark the beginning of each section with Section X, such as: Section 1\.?')
    remove(r'Mark the beginning of each section with Section 1 and Section 2\.?')
    remove(r'^\s*Section 2\.', None)
    remove(r'Use the markdown bullet points such as:\s*\* This is point 1\.?')
    remove(r'(?:your |the )?(?:ENTIRE )?(?:response|answer) should (?:also )?be in English, and in all lowercase letters\.?(?:\s*no capital letters are allowed\.?)?', lambda m:dict(kind='lower'))
    remove(r'(?:your |the )?(?:ENTIRE )?(?:response|answer) should (?:also )?be in English, and in all uppercase letters\.?(?:\s*no lowercase letters are allowed\.?)?', lambda m:dict(kind='upper'))
    remove(r'(?:your |the )?(?:response|answer) must contain a title, wrapped in double angular brackets, such as <<[^<>]+>>\.?', lambda m:dict(kind='title'))
    remove(r'At the end of your response, please explicitly add a postscript starting with P\.S\.', lambda m:dict(kind='postscript'))
    remove(r'Finish your response with this exact phrase ["\x27]([^"\x27]+)["\x27]\.?(?:\s*No other words should follow this phrase\.?)?', lambda m:dict(kind='ending',value=m[1]))
    # Restricted keyword syntax; preserve multiword phrases and reject placeholders.
    remove(r'Include (?:the following )?keywords:?\s*\[([^\[\]\n]+)\](?: in (?:the|your) response)?\.?',
           lambda m:dict(kind='keywords',values=[s.strip(' \"\x27') for s in m[1].split(',')]))
    remove(r'In your response, the word ["\x27\[]([^"\x27\]\n]+)["\x27\]] should appear (at least|exactly) (\d+) times\.?',
           lambda m:dict(kind='frequency',value=m[1],mode=m[2].lower(),n=int(m[3])))
    # A separate spelling of the same frequency requirement.
    remove(r'In your response, the word ([A-Za-z]+) should appear (at least|exactly) (\d+) times\.?',
           lambda m:dict(kind='frequency',value=m[1],mode=m[2].lower(),n=int(m[3])))
    remove(r'(?:your |the )?(?:response|answer) must have (\d+) paragraphs\.\s*Paragraphs are separated with the markdown divider:\s*\*\*\*',
           lambda m:dict(kind='paragraphs',mode='exactly',n=int(m[1])))
    task = re.sub(r'\s+', ' ', text).strip(' .\n')
    task = re.sub(r'^(?:Additionally|Also|Furthermore|Finally),\s*', '', task, flags=FLAGS)
    # No guessed or silently omitted constraints. Titles/examples don't count as a task.
    if re.search(r'\b(?:response|answer|keywords?|placeholders?|sentences?|bullet points?|paragraphs?|postscript|markdown|lowercase|uppercase|highlight|constraints?|format|section\s+\d|Section X|must|should|ensure|exactly|at least|at most|no more|do not|commas|quotation)\b|\[|\]|<<|>>|\*', task, FLAGS):
        return None
    if not 7 <= len(words(task)) <= 95 or not rules or len(rules)>4: return None
    if not re.search(r'\b(?:what|why|how|explain|describe|suggest|give|write|discuss|compare|outline|summarize|recommend|list|provide)\b',task,FLAGS): return None
    if any(r.get('n',0)>150 or (r['kind']=='sentences' and r['n']>6) or (r['kind']=='bullets' and r['n']>5) for r in rules): return None
    return task, rules


def verify(text, rules):
    checks = []
    for r in rules:
        k=r['kind'];n=r.get('n');mode=r.get('mode');ok=False
        if k=='words': ok=compare(len(text.split()),mode,n)
        elif k=='sentences': ok=compare(sentence_count(text),mode,n)
        elif k=='bullets': ok=compare(len(re.findall(r'(?m)^\s*[*-] +\S',text)),mode,n)
        elif k=='sections': ok=[int(v) for v in re.findall(r'(?im)^\s*section\s+(\d+)\s*[:.]?\s*$',text)]==list(range(1,n+1))
        elif k=='paragraphs': ok=len([p for p in re.split(r'(?m)^\s*\*\*\*\s*$',text) if p.strip()])==n
        elif k=='lower': ok=text==text.lower() and bool(re.search('[a-z]',text))
        elif k=='upper': ok=text==text.upper() and bool(re.search('[A-Z]',text))
        elif k=='title': ok=bool(re.match(r'^\s*<<[^<>\n]+>>\s*\n',text))
        elif k=='postscript': ok=bool(re.search(r'(?im)^\s*P\.S\.\s*\S.+$',text))
        elif k=='ending': ok=text.rstrip().endswith(r['value'])
        elif k=='keywords': ok=all(bool(re.search(r'(?<!\w)'+re.escape(v)+r'(?!\w)',text,FLAGS)) for v in r['values'])
        elif k=='frequency': ok=compare(len(re.findall(r'(?<!\w)'+re.escape(r['value'])+r'(?!\w)',text,FLAGS)),mode,n)
        else: raise ValueError(k)
        checks.append(bool(ok))
    return checks


def repetition(text):
    w=[s.lower() for s in words(text)]
    c=Counter(tuple(w[i:i+4]) for i in range(len(w)-3))
    return bool(c and max(c.values())>=3)


def fixtures():
    p='What are three ways to keep a shared garden tidy? Your response should contain exactly 3 bullet points. Use the markdown bullet points such as: * This is point 1.'
    task,rules=parse(p)
    good='* Put tools back after use.\n* Collect fallen litter each evening.\n* Remove weeds from the paths.'
    assert all(verify(good,rules)) and not all(verify('* Put tools away.',rules))
    assert not all(verify('',rules))
    assert parse('Your response should contain at least 3 sentences.') is None
    assert parse(p+' Do not use any commas.') is None
    for kind,good,bad in [('sections','Section 1\nTrees grow.\nSection 2\nLeaves fall.','Section 1\nTrees grow.'),('paragraphs','One.\n***\nTwo.','One.')]:
        assert all(verify(good,[dict(kind=kind,mode='exactly',n=2)]))
        assert not all(verify(bad,[dict(kind=kind,mode='exactly',n=2)]))
    assert sentence_count('Trees need sunlight. They use water. Leaves make food.')==3
    return dict(positive_negative_fixtures=True,empty_answers_fail=True,unscoped_prompts_rejected=True)
