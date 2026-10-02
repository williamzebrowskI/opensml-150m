"""Checks for leakage and task verifiers that can otherwise reward bad outputs."""
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from sft.grounded_rank_384.data import instruction_pass, instruction_rows, relation_rows, partition, ranking_row


class DataTests(unittest.TestCase):
    def test_scene_splits_are_disjoint(self):
        for generate in (instruction_rows, relation_rows):
            groups = {}
            for split in ('train', 'dev', 'test'):
                rows = generate(split, 96, 928384256)
                groups[split] = {r['group'] for r in rows}
                self.assertEqual(len(groups[split]), 96)
                self.assertTrue(all(partition(r['group']) == split for r in rows))
            self.assertFalse(groups['train'] & groups['dev'])
            self.assertFalse(groups['train'] & groups['test'])
            self.assertFalse(groups['dev'] & groups['test'])

    def test_json_rejects_duplicate_keys_and_extraneous_content(self):
        row = next(r for r in instruction_rows('train', 12, 41) if r['task'] == 'json')
        target = json.loads(row['answer'])
        duplicate = '{"person": "wrong", "person": ' + json.dumps(target['person']) + ', "day": ' + json.dumps(target['day']) + '}'
        self.assertFalse(instruction_pass(row, duplicate))
        self.assertFalse(instruction_pass(row, 'Here is the answer:\n' + row['answer']))
        target['extra'] = 'field'
        self.assertFalse(instruction_pass(row, json.dumps(target)))
        reordered = dict(reversed(list(json.loads(row['answer']).items())))
        self.assertTrue(instruction_pass(row, json.dumps(reordered)))

    def test_csv_target_sorts_the_actual_prompt_words(self):
        rows = [r for r in instruction_rows('train', 120, 71) if r['task'] == 'csv']
        for row in rows:
            inputs = row['prompt'].rsplit(': ', 1)[1].split(', ')
            self.assertEqual(row['answer'].split(','), sorted(inputs))
            self.assertFalse(instruction_pass(row, row['answer'] + '\nHere is the sorted list.'))

    def test_updated_invitation_preserves_other_content(self):
        for row in instruction_rows('train', 120, 91):
            if row['task'] != 'rewrite': continue
            updated_day = row['prompt'].split('updated day ', 1)[1].split(',', 1)[0]
            invitation = row['prompt'].split('Invitation: ', 1)[1]
            expected = invitation.rsplit(' on ', 1)[0] + ' on ' + updated_day + '.'
            self.assertEqual(row['answer'], expected)

    def test_video_group_stays_together_across_different_contexts(self):
        base = dict(activity_label='Gardening', ctx_a='A person digs a hole.', ctx_b='They',
                    endings=['plant a seed.', 'fly a kite.', 'swim away.', 'paint the sky.'],
                    label='0', ind=1, source_id='activitynet~fixture-video')
        other = dict(base, ctx_a='A person picks up a spade.', ind=2)
        left, right = ranking_row('hellaswag', base, 0), ranking_row('hellaswag', other, 1)
        self.assertNotEqual(left['id'], right['id'])
        self.assertEqual(left['group'], right['group'])
        self.assertEqual(left['split'], right['split'])


if __name__ == '__main__': unittest.main()
