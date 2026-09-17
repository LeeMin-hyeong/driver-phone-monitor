import unittest

from binary_policy import decide
from evaluate_dataset import summarize


class BinaryPolicyTests(unittest.TestCase):
    def test_threshold_equality_and_zero_score(self):
        policy = dict(threshold=.5, unscored_class='positive')
        self.assertEqual(decide(.5, policy), 'positive')
        self.assertEqual(decide(.49, policy), 'negative')
        self.assertEqual(decide(0, policy), 'negative')
        self.assertEqual(decide(None, policy), 'positive')

    def test_negative_fallback(self):
        self.assertEqual(decide(None, dict(threshold=.5, unscored_class='negative')), 'negative')

    def test_nonfinite_is_error(self):
        with self.assertRaises(ValueError):
            decide(float('nan'), dict(threshold=.5, unscored_class='positive'))

    def test_unscored_cases_count_in_binary_metrics(self):
        rows = [dict(target=1, score=.1, status='negative'),
                dict(target=1, score=None, status='positive', reason='driver_pose_missing'),
                dict(target=0, score=None, status='positive', reason='driver_pose_missing'),
                dict(target=0, score=.1, status='negative')]
        result = summarize(rows, binary=True)
        self.assertEqual(result['overall']['n'], 4)
        self.assertEqual(result['overall']['confusion_matrix'], [[1, 1], [1, 1]])
        self.assertEqual(result['unknown'], 0)
        self.assertEqual(result['unscored'], 2)
        self.assertEqual(result['coverage'], 1)


if __name__ == '__main__':
    unittest.main()
