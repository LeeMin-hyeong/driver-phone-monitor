"""Automatic driver selection and body-relative, filename-independent features.

Default camera convention: front-facing cabin photo with driver on image right.
Use driver_side='left' for mirrored/right-hand-drive views.
"""
import math
import numpy as np

VERSION = 4


def driver_person_box(boxes, side):
    """Use independently detected people to avoid treating the only pose as driver."""
    people=[b for b in boxes if b['cls']==0 and b['confidence']>=0.25]
    if not people:
        return None
    def area(b):
        x1,y1,x2,y2=b['box']
        return max(0,x2-x1)*max(0,y2-y1)
    largest=max(area(b) for b in people)
    bottom=max(b['box'][3] for b in people if area(b)>=largest*.25)
    front=[b for b in people if area(b)>=largest*.25 and b['box'][3]>=bottom-.22]
    sign=1 if side=='right' else -1
    return max(front,key=lambda b:sign*(b['box'][0]+b['box'][2])/2)['box']


def features(observation, driver_side='right'):
    aspect = observation['width'] / observation['height']
    poses = [np.asarray(p, dtype=float) for p in observation['poses']]
    valid = [i for i, p in enumerate(poses) if min(p[11, 3], p[12, 3]) >= 0.4]
    if not valid:
        return None, dict(status='unknown', reason='no_reliable_body_landmarks')
    def xy(p):
        return p[..., :2] * [aspect, 1]
    scales = [max(np.linalg.norm(xy(p[11]) - xy(p[12])), 0.03) for p in poses]
    # Multi-pose inference occasionally returns two skeletons for one person.
    keep = []
    for i in sorted(valid, key=lambda i: float(poses[i][[11,12,13,14,15,16],3].mean()), reverse=True):
        center_i = xy(poses[i][[11,12]]).mean(axis=0)
        if not any(np.linalg.norm(center_i-xy(poses[j][[11,12]]).mean(axis=0)) < 0.4*max(scales[i],scales[j]) for j in keep):
            keep.append(i)
    valid = keep
    seat_box=driver_person_box(observation['boxes'],driver_side)
    if seat_box is None:
        return None,dict(status='unknown',reason='driver_person_not_detected')
    x1,y1,x2,y2=seat_box
    seat_matches=[]
    for i in valid:
        cx,cy=poses[i][[11,12],:2].mean(axis=0)
        if x1<=cx<=x2 and y1<=cy<=y2:
            cost=abs(cx-(x1+x2)/2)/max(x2-x1,.01)+.3*abs(cy-(y1+.3*(y2-y1)))/max(y2-y1,.01)
            seat_matches.append((float(cost),i))
    if not seat_matches:
        return None,dict(status='unknown',reason='driver_pose_missing',driver_box=seat_box)
    seat_matches.sort()
    if len(seat_matches)>1 and seat_matches[1][0]-seat_matches[0][0]<.15:
        return None,dict(status='unknown',reason='driver_pose_ambiguous',driver_box=seat_box)
    ranked=list(valid)
    driver = seat_matches[0][1]
    # Ranking now follows independently detected seat occupancy, not pose count.
    ranked=[driver]+[i for i in ranked if i!=driver]
    p = poses[driver]
    scale = scales[driver]
    center = (xy(p[11]) + xy(p[12])) / 2
    axis = (xy(p[12])-xy(p[11]))/scale
    basis = np.asarray([axis, [-axis[1],axis[0]]])
    f = {}
    # Coordinates relative to shoulders avoid learning room/background location.
    for j in (0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 19, 20, 23, 24):
        pos = basis @ ((xy(p[j]) - center) / scale)
        f[f'p{j}_x'], f[f'p{j}_y'] = np.clip(pos, -4, 4)
        f[f'p{j}_visibility'] = p[j, 3]
    for name, a, b, c in (('left_elbow', 11, 13, 15), ('right_elbow', 12, 14, 16)):
        u, v = xy(p[a]) - xy(p[b]), xy(p[c]) - xy(p[b])
        f[name] = float(np.dot(u, v) / max(np.linalg.norm(u)*np.linalg.norm(v), 1e-6))
    def rect_distance(point, box):
        x1, y1, x2, y2 = np.asarray(box) * [aspect, 1, aspect, 1]
        return math.hypot(max(x1-point[0], 0, point[0]-x2), max(y1-point[1], 0, point[1]-y2)) / scale
    phones = [b for b in observation['boxes'] if b['cls'] == 67]
    evidence = []
    for wrist in (15, 16):
        wp = xy(p[wrist])
        pairs = []
        for b in phones:
            d = rect_distance(wp, b['box'])
            others = [rect_distance(xy(q[j]), b['box']) for i, q in enumerate(poses) if i != driver and i in valid
                      for j in (15, 16) if q[j, 3] >= 0.4]
            competitor = min(others, default=10)
            pairs.append((d, b['confidence'], competitor))
        nearest = max(pairs, key=lambda v: v[1]*math.exp(-4*v[0]), default=(5, 0, 5))
        f[f'w{wrist}_phone_distance'] = min(nearest[0], 5)
        f[f'w{wrist}_phone_confidence'] = nearest[1]
        f[f'w{wrist}_ownership_margin'] = np.clip(nearest[2] - nearest[0], -5, 5)
        f[f'w{wrist}_phone_contact'] = nearest[1]*math.exp(-4*nearest[0])
        for cls,name in ((39,'bottle'),(41,'cup')):
            f[f'w{wrist}_{name}_contact'] = max((b['confidence']*math.exp(-4*rect_distance(wp,b['box']))
                                                for b in observation['boxes'] if b['cls']==cls),default=0)
        f[f'w{wrist}_nose_distance'] = float(np.linalg.norm(wp-xy(p[0]))/scale)
        matches = []
        for h in observation['hands']:
            h = np.asarray(h)
            d = np.linalg.norm(xy(h[0]) - wp) / scale
            other_wrists = [np.linalg.norm(xy(h[0]) - xy(q[j])) / scales[i]
                            for i, q in enumerate(poses) for j in (15, 16)
                            if i in valid and (i != driver or j != wrist) and q[j, 3] >= 0.4]
            if d < 0.45 and d + 0.05 < min(other_wrists, default=10):
                matches.append((d, h))
        hand = min(matches, key=lambda item: item[0])[1] if matches else None
        f[f'w{wrist}_hand_found'] = int(hand is not None)
        for j in (4, 8, 12, 16, 20):
            # Finger tip distance from wrist measures compact grip vs open palm.
            f[f'w{wrist}_finger{j}'] = float(np.linalg.norm(xy(hand[j])-xy(hand[0]))/scale) if hand is not None else -1
        hand_pairs=[]
        if hand is not None:
            palm=xy(hand[[0,5,9,13,17]]).mean(axis=0)
            for phone in phones:
                palm_d=rect_distance(palm,phone['box'])
                tip_ds=[rect_distance(xy(hand[j]),phone['box']) for j in (4,8,12,16,20)]
                contact_d=min(palm_d,sum(sorted(tip_ds)[:2])/2)
                contact_score=phone['confidence']*math.exp(-4*contact_d)
                other_ds=[rect_distance(xy(poses[i][j]),phone['box']) for i in valid if i!=driver
                          for j in (15,16) if poses[i][j,3]>=.4]
                hand_pairs.append((contact_score,palm_d,tip_ds,phone['confidence'],min(other_ds,default=10)-contact_d))
        best_hand=max(hand_pairs,key=lambda item:item[0],default=(0,5,[5]*5,0,0))
        f[f'w{wrist}_palm_phone_distance']=min(best_hand[1],5)
        for j,d in zip((4,8,12,16,20),best_hand[2]):
            f[f'w{wrist}_tip{j}_phone_distance']=min(d,5)
        f[f'w{wrist}_hand_phone_contact']=best_hand[0]
        f[f'w{wrist}_hand_phone_confidence']=best_hand[3]
        f[f'w{wrist}_hand_ownership_margin']=float(np.clip(best_hand[4],-5,5))
        evidence.append(dict(wrist=wrist, phone_distance=nearest[0], phone_confidence=nearest[1],
                             competitor_distance=nearest[2], hand_found=hand is not None,
                             palm_phone_distance=best_hand[1],tip_phone_distances=best_hand[2],
                             hand_phone_contact=best_hand[0]))
    return f, dict(driver_index=driver, driver_candidates=ranked,driver_box=seat_box,
                   driver_margin=seat_matches[1][0]-seat_matches[0][0] if len(seat_matches)>1 else 1,
                   evidence=evidence, status='features_extracted')
