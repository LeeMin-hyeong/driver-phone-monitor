import unittest
from evaluate_dataset import summarize


class EvaluationTests(unittest.TestCase):
    def test_abstention_denominator_and_confusion_matrix(self):
        rows=[dict(target=1,score=.9,status='positive'),
              dict(target=0,score=.1,status='negative'),
              dict(target=0,score=.55,status='unknown'),
              dict(target=1,score=.45,status='unknown'),
              dict(target=1,score=None,status='unknown')]
        result=summarize(rows)
        self.assertEqual(result['overall']['confusion_matrix'],[[1,1],[1,1]])
        self.assertEqual(result['overall']['f1'],.5)
        self.assertEqual(result['selective']['accuracy'],1)
        self.assertEqual(result['coverage'],.4)
        self.assertEqual(result['extraction_failures'],1)

    def test_empty_subset(self):
        result=summarize([])
        self.assertIsNone(result['overall'])
        self.assertIsNone(result['coverage'])

    def test_driver_guard_not_counted_as_extractor_crash(self):
        result=summarize([dict(target=1,score=None,status='unknown',reason='driver_pose_missing')])
        self.assertEqual(result['extraction_failures'],0)
        self.assertEqual(result['unscored'],1)
        self.assertEqual(result['unknown'],1)
        self.assertEqual(result['unscored_reasons'],{'driver_pose_missing':1})


if __name__=='__main__':
    unittest.main()
