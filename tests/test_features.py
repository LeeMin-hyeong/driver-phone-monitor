import unittest
from features import features


def body(x, y=0.4, size=0.2):
    p = [[x,y,0,1] for _ in range(33)]
    for j,dx,dy in ((11,-.5,0),(12,.5,0),(13,-.6,.5),(14,.6,.5),(15,-.7,1),(16,.7,1)):
        p[j]=[x+dx*size,y+dy*size,0,1]
    return p


def observation(poses, boxes=None):
    people=[]
    for p in poses:
        x=(p[11][0]+p[12][0])/2;y=p[11][1];s=p[12][0]-p[11][0]
        people.append(dict(cls=0,confidence=.9,box=[x-.65*s,y-.5*s,x+.65*s,max(y+2*s,.85 if y>=.3 else 0)]))
    return dict(width=1000,height=1000,poses=poses,boxes=people+(boxes or []),hands=[])


class AutomaticDriverTests(unittest.TestCase):
    def test_no_body(self):
        f, info=features(observation([]))
        self.assertIsNone(f)
        self.assertEqual(info['status'],'unknown')

    def test_right_driver_with_rear_passenger(self):
        _,info=features(observation([body(.25),body(.65),body(.9,.1,.07)]))
        self.assertEqual(info['driver_index'],1)

    def test_left_driver(self):
        _,info=features(observation([body(.25),body(.65)]),'left')
        self.assertEqual(info['driver_index'],0)

    def test_driver_smaller_than_passenger(self):
        _,info=features(observation([body(.3,.4,.3),body(.7,.4,.16)]))
        self.assertEqual(info['driver_index'],1)

    def test_duplicate_pose_removed(self):
        _,info=features(observation([body(.3),body(.7),body(.705)]))
        self.assertEqual(len(info['driver_candidates']),2)

    def test_passenger_phone_does_not_touch_driver(self):
        phone=dict(cls=67,box=[.09,.58,.14,.65],confidence=.9)
        f,_=features(observation([body(.25),body(.7)],[phone]))
        self.assertLess(f['w15_phone_contact'],.01)

    def test_cup_is_separate_from_phone(self):
        cup=dict(cls=41,box=[.54,.58,.59,.65],confidence=.9)
        f,_=features(observation([body(.7)],[cup]))
        self.assertEqual(f['w15_phone_contact'],0)
        self.assertGreater(f['w15_cup_contact'],.8)

    def test_missing_driver_pose_does_not_select_passenger(self):
        o=observation([body(.25)])
        o['boxes'].append(dict(cls=0,confidence=.9,box=[.5,.2,.9,.9]))
        f,info=features(o)
        self.assertIsNone(f)
        self.assertEqual(info['reason'],'driver_pose_missing')

    def test_missing_left_driver_pose(self):
        o=observation([body(.7)])
        o['boxes'].append(dict(cls=0,confidence=.9,box=[.05,.2,.4,.9]))
        f,info=features(o,'left')
        self.assertIsNone(f)
        self.assertEqual(info['reason'],'driver_pose_missing')

    def test_no_person_detection_is_unknown(self):
        o=observation([body(.7)]);o['boxes']=[]
        self.assertEqual(features(o)[1]['reason'],'driver_person_not_detected')

    def test_fingertips_capture_phone_far_from_wrist(self):
        phone=dict(cls=67,confidence=.9,box=[.64,.57,.68,.61])
        o=observation([body(.7)],[phone])
        h=[[.56,.6,0] for _ in range(21)]
        for j in (5,9,13,17):h[j]=[.62,.59,0]
        for j in (4,8,12,16,20):h[j]=[.66,.59,0]
        o['hands']=[h]
        f,info=features(o)
        self.assertGreater(f['w15_phone_distance'],.3)
        self.assertEqual(f['w15_tip8_phone_distance'],0)
        self.assertLess(f['w15_palm_phone_distance'],f['w15_phone_distance'])
        self.assertGreater(f['w15_hand_phone_contact'],f['w15_phone_contact'])

    def test_passenger_fingers_not_used_for_driver(self):
        o=observation([body(.25),body(.7)],[dict(cls=67,confidence=.9,box=[.1,.58,.14,.62])])
        o['hands']=[[[.11,.6,0] for _ in range(21)]]
        f,_=features(o)
        self.assertEqual(f['w15_hand_found'],0)
        self.assertEqual(f['w15_hand_phone_contact'],0)
        self.assertEqual(f['w15_tip8_phone_distance'],5)


if __name__=='__main__':
    unittest.main()
