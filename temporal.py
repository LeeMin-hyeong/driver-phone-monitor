"""Time-weighted smoothing of final binary predictions, independent of models."""
from collections import deque
import math


class PredictionSmoother:
    window_seconds = 1.0
    positive_ratio_threshold = 0.8
    release_seconds = 0.7
    max_gap_seconds = 0.5

    def __init__(self):
        self.reset()

    def reset(self):
        self.samples = deque()
        self.started = None
        self.negative_since = None
        self.alert = False

    def update(self, positive, timestamp):
        if not math.isfinite(timestamp):
            raise ValueError('Timestamp must be finite')
        positive = bool(positive)
        if self.samples and (timestamp < self.samples[-1][0] or
                             timestamp-self.samples[-1][0] > self.max_gap_seconds):
            self.reset()
        if self.started is None:
            self.started = timestamp
        # A sample represents the interval until the next observed frame.
        if self.samples and timestamp == self.samples[-1][0]:
            self.samples.pop()
        self.samples.append((timestamp, positive))
        cutoff = timestamp-self.window_seconds
        while len(self.samples) > 1 and self.samples[1][0] <= cutoff:
            self.samples.popleft()
        positive_seconds = 0.
        for (start,value),(end,_) in zip(self.samples,list(self.samples)[1:]):
            if value:
                positive_seconds += max(0.,end-max(start,cutoff))
        observed = min(self.window_seconds,timestamp-self.started)
        ratio = positive_seconds/observed if observed > 0 else 0.
        if positive:
            self.negative_since = None
        elif self.negative_since is None:
            self.negative_since = timestamp
        negative_seconds = (timestamp-self.negative_since
                            if self.negative_since is not None else 0.)
        if self.alert:
            if negative_seconds >= self.release_seconds-1e-9:
                self.alert = False
        elif observed >= self.window_seconds-1e-9 and ratio >= self.positive_ratio_threshold-1e-9:
            self.alert = True
        return dict(prediction=int(self.alert),positive_ratio=ratio,
                    observed_seconds=observed,negative_seconds=negative_seconds)
