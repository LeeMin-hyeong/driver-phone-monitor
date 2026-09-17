"""Pure geometry rules. Coordinates use pixels; thresholds use shoulder width."""
from dataclasses import dataclass
from math import hypot


@dataclass
class Point:
    x: float
    y: float
    confidence: float = 1.0


def distance(a, b):
    return hypot(a.x - b.x, a.y - b.y)


def box_distance(p, box):
    x1, y1, x2, y2 = box
    return hypot(max(x1 - p.x, 0, p.x - x2), max(y1 - p.y, 0, p.y - y2))


def classify(poses, hands, phones, config, width, height):
    """Return evidence, not a calibrated probability or proof of physical grip."""
    roi = config['driver_shoulder_roi']
    minimum = config.get('landmark_confidence', 0.6)
    scales = [distance(p[11], p[12]) for p in poses]
    candidates = []
    for i, p in enumerate(poses):
        cx, cy = (p[11].x + p[12].x) / (2 * width), (p[11].y + p[12].y) / (2 * height)
        if (min(p[11].confidence, p[12].confidence) >= minimum
                and roi[0] <= cx <= roi[2] and roi[1] <= cy <= roi[3]
                and config['min_shoulder_width_ratio'] <= scales[i] / width <= config['max_shoulder_width_ratio']):
            candidates.append(i)
    def result(status, reason, evidence=None):
        return dict(status=status, reason=reason, driver_index=candidates[0] if len(candidates) == 1 else None,
                    evidence=evidence or [])
    if len(candidates) != 1:
        return result('unknown', 'driver_missing_or_ambiguous')
    driver = candidates[0]
    # Assign each hand to a unique pose wrist; handedness is not seat identity.
    assigned = []
    for hand in hands:
        matches = []
        for i, pose in enumerate(poses):
            for shoulder, elbow, wrist in ((11, 13, 15), (12, 14, 16)):
                if scales[i] < 1 or min(pose[j].confidence for j in (shoulder, elbow, wrist)) < minimum:
                    continue
                d = distance(hand[0], pose[wrist]) / scales[i]
                if d <= config.get('hand_wrist_distance', 0.25):
                    matches.append((d, i, wrist))
        matches.sort()
        unique = bool(matches) and (len(matches) == 1 or matches[1][0] - matches[0][0] >= 0.10)
        assigned.append((hand, matches[0][1] if unique else None))
    evidence = []
    ambiguous = False
    for phone in phones:
        box = phone['box']
        owners = set()
        unassigned_contact = False
        for hand, owner in assigned:
            scale = scales[owner] if owner is not None else scales[driver]
            # Require palm proximity plus multiple fingers near the phone rectangle.
            palm = Point(sum(hand[j].x for j in (0, 5, 9, 13, 17)) / 5,
                         sum(hand[j].y for j in (0, 5, 9, 13, 17)) / 5)
            near = sum(box_distance(hand[j], box) <= 0.08 * scale for j in (4, 8, 12, 16, 20))
            if box_distance(palm, box) <= 0.16 * scale and near >= 2:
                if owner is None:
                    unassigned_contact = True
                else:
                    owners.add(owner)
        # Even a competitor whose fingers are occluded must veto confident ownership.
        competitors = set()
        for i, pose in enumerate(poses):
            for wrist in (15, 16):
                if pose[wrist].confidence >= minimum and box_distance(pose[wrist], box) <= 0.25 * scales[i]:
                    competitors.add(i)
        confirmed = owners == {driver} and not unassigned_contact and not (competitors - {driver})
        evidence.append(dict(phone=phone, owners=sorted(owners), nearby_people=sorted(competitors),
                             unassigned_contact=unassigned_contact, driver_contact=confirmed))
        ambiguous |= unassigned_contact or driver in competitors or driver in owners
    if any(e['driver_contact'] for e in evidence):
        return result('holding', 'driver_hand_phone_contact', evidence)
    if ambiguous:
        return result('unknown', 'occlusion_or_ownership_ambiguous', evidence)
    if any(poses[driver][j].confidence < minimum for j in (13, 14, 15, 16)):
        return result('unknown', 'driver_arms_not_visible', evidence)
    if not all(any(owner == driver and distance(h[0], poses[driver][j]) < 0.25 * scales[driver]
                   for h, owner in assigned) for j in (15, 16)):
        return result('unknown', 'driver_hands_not_visible', evidence)
    return result('not_holding', 'no_detected_driver_phone_contact', evidence)
