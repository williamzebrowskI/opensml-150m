"""Exercise the evaluation path without model weights or optimizer updates."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import mlx.core as mx
import launch

class EvaluationTests(unittest.TestCase):
    def test_every_turn_constraints_and_cached_receipt(self):
        class Model:
            def eval(self):pass
            def __call__(self,x):return {'logits':mx.zeros((1,x.shape[1],4))}
        backend=SimpleNamespace(model=Model(),pad=0,tokenizer=None)
        chat=dict(id='chat',source='chat',messages=[dict(role='system',content='Be helpful.'),dict(role='user',content='First question'),dict(role='assistant',content='Reference one'),dict(role='user',content='Second question'),dict(role='assistant',content='Reference two')])
        ins=dict(id='instruction',source='instructions',messages=[dict(role='user',content='Use no commas.'),dict(role='assistant',content='Trees need water.')],checks=[dict(family='no_comma')])
        cfg=dict(training_counts={'chat':1,'instructions':1},generation_per_source={'chat':1,'instructions':1},validation_conversations=2,context=2048,max_new_tokens=384)
        encoded=[dict(x=[2,3],y=[-100,1],targets=1)]
        histories=[]
        def generate(b,history,limit):
            histories.append(list(history))
            return dict(text='Trees need water.',tokens=[2,3],stop='eos')
        with tempfile.TemporaryDirectory(prefix='public-repair-eval-') as tmp, patch.object(launch,'RUN',Path(tmp)), patch('data.turns',return_value=encoded), patch('sft.text_followup_512_v1.engine.assess',return_value={'metrics':{},'answers':[]}), patch('sft.conversation_foundation_v1.engine.generate',side_effect=generate):
            launch.evaluate(backend,{'dev':[chat,ins]},cfg,0,'parent',{'test':'contract'})
            result=json.loads((Path(tmp)/'evaluations/update_00000.json').read_text())
            metrics=result['public_development']['metrics']
            self.assertEqual(metrics['generated_turns'],3)
            self.assertEqual(metrics['validation_conversations'],2)
            self.assertEqual(metrics['verified_named_constraint_passes'],1)
            self.assertEqual(histories[1][-2]['content'],'Trees need water.')
            self.assertEqual(histories[0][0]['role'],'system')
            launch.evaluate(backend,{'dev':[chat,ins]},cfg,0,'parent',{'test':'contract'})
            self.assertEqual(len(histories),3)

if __name__=='__main__':unittest.main()
