"""Binary decisions, including an explicit fallback when no score is available."""
import math


def decide(score, policy):
    if score is None:
        return policy['unscored_class']
    if not math.isfinite(score):
        raise ValueError('Classifier score must be finite')
    return 'positive' if score >= policy['threshold'] else 'negative'


DEFAULT_POLICY = dict(threshold=0.5, unscored_class='negative')
