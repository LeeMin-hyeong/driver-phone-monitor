import unittest
from legacy.logic import Point, classify


def pose(x):
    p = [Point(x, 100) for _ in range(33)]
    for j, dx, y in ((11, -50, 100), (12, 50, 100), (13, -50, 150),
                     (14, 50, 150), (15, -50, 200), (16, 50, 200)):
        p[j] = Point(x + dx, y)
    return p


def hand(x):
    return [Point(x + (i % 4), 200 + (i % 5)) for i in range(21)]


class OwnershipTests(unittest.TestCase):
    def setUp(self):
        self.poses = [pose(200), pose(650)]
        self.hands = [hand(150), hand(250), hand(600), hand(700)]
        self.config = dict(driver_shoulder_roi=[0.1, 0.05, 0.3, 0.3],
                           min_shoulder_width_ratio=0.08, max_shoulder_width_ratio=0.15)

    def run_case(self, x=None):
        phones = [] if x is None else [dict(box=[x-5, 195, x+10, 220], confidence=0.9)]
        return classify(self.poses, self.hands, phones, self.config, 1000, 500)['status']

    def test_driver(self):
        self.assertEqual(self.run_case(150), 'holding')

    def test_passenger(self):
        self.assertEqual(self.run_case(600), 'not_holding')

    def test_rear_passenger_in_roi(self):
        rear = pose(200)
        rear[11], rear[12] = Point(180, 100), Point(220, 100)
        self.poses = [rear]
        self.assertEqual(self.run_case(), 'unknown')

    def test_two_driver_candidates(self):
        self.poses.append(pose(210))
        self.assertEqual(self.run_case(150), 'unknown')

    def test_missing_driver(self):
        self.poses = [pose(650)]
        self.assertEqual(self.run_case(), 'unknown')

    def test_hidden_hand(self):
        self.hands = []
        self.assertEqual(self.run_case(150), 'unknown')

    def test_competing_wrist(self):
        self.poses[1][15] = Point(150, 200)
        self.assertEqual(self.run_case(150), 'unknown')

    def test_low_visibility(self):
        self.poses[0][15].confidence = 0.1
        self.assertEqual(self.run_case(), 'unknown')

    def test_no_phone(self):
        self.assertEqual(self.run_case(), 'not_holding')

    def test_console_phone(self):
        self.assertEqual(self.run_case(420), 'not_holding')


if __name__ == '__main__':
    unittest.main()
