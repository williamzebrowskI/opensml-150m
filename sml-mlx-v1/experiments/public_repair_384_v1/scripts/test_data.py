"""Regression checks for ambiguous prompts and invalid public targets."""
import unittest
from data import parse_checks,verify

class ConstraintTests(unittest.TestCase):
    def test_mislabeled_capitalization(self):
        self.assertIsNone(parse_checks('Start every sentence with a capital letter.', ['in english and capital']))

    def test_ranges_and_local_counts_rejected(self):
        label=['length constraints:number of sentences']
        self.assertIsNone(parse_checks('Use 3-5 sentences.',label))
        self.assertIsNone(parse_checks('Use two sentences in each paragraph.',label))
        self.assertIsNone(parse_checks('Use two sentences, then three sentences.',label))

    def test_word_count_and_comma_both_required(self):
        checks=parse_checks('Use exactly four words without commas.', ['length constraints:number of words','punctuation:use no comma'])
        self.assertEqual(verify('Trees need clean water',checks),[True,True])
        self.assertEqual(verify('Trees, need clean water',checks),[True,False])
        self.assertEqual(verify('Trees need water',checks),[False,True])

    def test_case_preserves_full_constraint(self):
        checks=parse_checks('Write your entire answer in uppercase.', ['in english and capital'])
        self.assertEqual(verify('TREES NEED WATER.',checks),[True])
        self.assertEqual(verify('Trees NEED WATER.',checks),[False])

    def test_end_phrase_exact(self):
        checks=parse_checks('Finish with the exact phrase "Thank you."', ['specific ending'])
        self.assertEqual(verify('Trees need water. Thank you.',checks),[True])
        self.assertEqual(verify('Trees need water. thank you.',checks),[False])
        self.assertEqual(verify('Thank you. Extra words.',checks),[False])

    def test_quoted_ending_composition(self):
        checks=parse_checks('Wrap the whole response in double quotes. Finish with "Thank you."', ['use quotation','specific ending'])
        self.assertEqual(verify('"Trees need water. Thank you."',checks),[True,True])

    def test_explicit_paragraph_delimiter(self):
        checks=parse_checks('Use exactly two paragraphs separated by ***.', ['length constraints:number of paragraphs'])
        self.assertEqual(verify('Trees need water.\n***\nPlants need sunlight.',checks),[True])
        self.assertEqual(verify('Trees need water.\n\nPlants need sunlight.',checks),[False])

    def test_unsupported_metadata_rejected(self):
        self.assertIsNone(parse_checks('Write a poem.', ['unknown requirement']))

if __name__=='__main__':unittest.main()
