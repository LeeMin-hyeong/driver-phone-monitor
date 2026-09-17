import unittest
from data_labels import resolve_labels


class LabelTests(unittest.TestCase):
    def test_supported_names_keep_positive_semantics(self):
        for pos,neg in [('phone','normal'),('들고있음','안들고있음'),('보고있음','안보고있음')]:
            self.assertEqual(resolve_labels([neg,pos]),(pos,neg))

    def test_ambiguous_directories_require_explicit_selection(self):
        names=['phone','normal','들고있음','안들고있음']
        with self.assertRaises(ValueError):
            resolve_labels(names)
        self.assertEqual(resolve_labels(names,'phone','normal'),('phone','normal'))

    def test_missing_or_partial_labels_fail(self):
        with self.assertRaises(ValueError):
            resolve_labels(['phone'])
        with self.assertRaises(ValueError):
            resolve_labels(['phone','normal'], positive='phone')
