"""Resolve supported folder-label names without changing class semantics."""
LABEL_PAIRS = [('phone', 'normal'), ('들고있음', '안들고있음'), ('보고있음', '안보고있음')]


def resolve_labels(labels, positive=None, negative=None):
    available = set(labels)
    if positive is not None or negative is not None:
        if not positive or not negative or positive == negative:
            raise ValueError('Specify both distinct --positive and --negative labels')
        if not {positive, negative} <= available:
            raise ValueError('Requested label folders are missing')
        return positive, negative
    pairs = [pair for pair in LABEL_PAIRS if set(pair) <= available]
    if len(pairs) != 1:
        raise ValueError('Cannot infer labels; specify --positive and --negative')
    return pairs[0]
