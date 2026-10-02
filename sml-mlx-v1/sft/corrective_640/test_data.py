"""Check composed targets, leakage boundaries, and complete conversation masks."""
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from sml_v1.common import read_json
from sml_v1.tokenization import Tokenizer
from sft.corrective_640.data import instruction_rows, instruction_pass, authored_replay, authored_grounded, partition
from sft.intact_smoltalk_base_pilot.data import turns, visible_prefix


class DataTests(unittest.TestCase):
    def test_scenes_and_presentation_variants(self):
        for generate in (instruction_rows, authored_grounded, authored_replay):
            groups = []
            for split in ('train', 'dev', 'test'):
                rows = generate(split, 64, 928640896)
                self.assertTrue(all(partition(r['group']) == split for r in rows))
                groups.append({r['group'] for r in rows})
                self.assertEqual(len(groups[-1]), 64)
                if generate is instruction_rows:
                    self.assertTrue(all(r['variant'] in ({0, 1} if split == 'train' else {2} if split == 'dev' else {3}) for r in rows))
            self.assertFalse(groups[0] & groups[1] | groups[0] & groups[2] | groups[1] & groups[2])

    def test_composed_constraints_reject_partial_compliance(self):
        for row in instruction_rows('train', 64, 71):
            self.assertTrue(instruction_pass(row, row['answer']))
            self.assertFalse(instruction_pass(row, row['answer'] + '\nExplanation: done.'))
            if row['task'] == 'json-sorted-upper':
                target = json.loads(row['answer'])
                target['words'] = [w.lower() for w in target['words']]
                self.assertFalse(instruction_pass(row, json.dumps(target)))
                target = json.loads(row['answer'])
                target['words'].reverse()
                self.assertFalse(instruction_pass(row, json.dumps(target)))
            if row['task'] == 'json-updated-lower':
                target = json.loads(row['answer']); target['extra'] = 'bad'
                self.assertFalse(instruction_pass(row, json.dumps(target)))
                duplicate = '{"day":"wrong",' + row['answer'][1:]
                self.assertFalse(instruction_pass(row, duplicate))
            if row['task'] == 'six-words-lower':
                self.assertEqual(len(row['answer'].split()), 6)
                self.assertFalse(instruction_pass(row, row['answer'].upper()))

    def test_every_assistant_turn_is_supervised_without_user_or_history_targets(self):
        tok = Tokenizer(ROOT / 'tokenizer/bytebpe32k_v1')
        for row in authored_replay('train', 8, 91):
            encoded = turns(tok, row, 1024)
            self.assertEqual([r['message_index'] for r in encoded], [1, 3])
            for r in encoded:
                i = r['message_index']; prefix = tok.encode(visible_prefix(row['messages'][:i]))
                ids = tok.encode(visible_prefix(row['messages'][:i]) + ' ' + row['messages'][i]['content']) + [tok.eos]
                self.assertEqual(r['y'][:len(prefix) - 1], [-100] * (len(prefix) - 1))
                self.assertEqual(r['y'][len(prefix) - 1:], ids[len(prefix):])
                self.assertEqual(r['y'][-1], tok.eos)

    def test_prepared_excludes_previous_round_groups_and_reserved_splits(self):
        path = Path(__file__).parent / 'prepared.json'
        self.assertTrue(path.exists(), 'Run --prepare before checking the selected data')
        data = read_json(path)
        old = read_json(ROOT / 'sft/grounded_rank_384/prepared.json')
        old_groups = {r['group'] for s in ('train', 'dev', 'test') for rows in old[s].values() for r in rows}
        groups = [{r['group'] for rows in data[s].values() for r in rows} for s in ('train', 'dev', 'test')]
        for g in groups: self.assertFalse(g & old_groups)
        self.assertFalse(groups[0] & groups[1] | groups[0] & groups[2] | groups[1] & groups[2])
        self.assertEqual(len(data['train']['grounded']), 1024)
        self.assertEqual(len(data['train']['instruction']), 1280)
        self.assertEqual(len(data['train']['replay']), 512)
        self.assertEqual(len(data['train']['ranking']), 1024)


if __name__ == '__main__': unittest.main()
