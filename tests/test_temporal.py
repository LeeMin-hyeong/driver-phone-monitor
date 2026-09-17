import unittest
from temporal import PredictionSmoother


class SmoothingTests(unittest.TestCase):
    def test_startup_and_exact_boundary(self):
        s = PredictionSmoother()
        for i in range(10):
            self.assertEqual(s.update(True,i/10)['prediction'],0)
        self.assertEqual(s.update(True,1.)['prediction'],1)

    def test_short_miss_is_tolerated(self):
        s = PredictionSmoother()
        for i in range(11):
            result = s.update(i != 5,i/10)
        self.assertEqual(result['prediction'],1)
        self.assertAlmostEqual(result['positive_ratio'],.9)
        self.assertEqual(s.update(False,1.1)['prediction'],1)
        self.assertEqual(s.update(True,1.2)['prediction'],1)

    def test_release_after_continuous_negative(self):
        s = PredictionSmoother()
        for i in range(11): s.update(True,i/10)
        for i in range(11,18):
            self.assertEqual(s.update(False,i/10)['prediction'],1)
        self.assertEqual(s.update(False,1.8)['prediction'],0)

    def test_duration_not_frame_count(self):
        s = PredictionSmoother()
        s.update(True,0.)
        s.update(False,.1)
        s.update(False,.5)
        for i in range(60,101): result = s.update(True,i/100)
        self.assertAlmostEqual(result['positive_ratio'],.5)
        self.assertEqual(result['prediction'],0)

    def test_gap_and_rewind_reset(self):
        for timestamp in (3.,.2):
            s = PredictionSmoother()
            for i in range(11): s.update(True,i/10)
            result = s.update(True,timestamp)
            self.assertEqual(result['prediction'],0)
            self.assertEqual(result['observed_seconds'],0.)

    def test_duplicate_timestamp_adds_no_time(self):
        s = PredictionSmoother()
        for _ in range(100): result = s.update(True,0.)
        self.assertEqual(result['prediction'],0)
        self.assertEqual(result['observed_seconds'],0.)


if __name__ == '__main__': unittest.main()
