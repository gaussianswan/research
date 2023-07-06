def label_by_threshold(val: float, upper_threshold: float = 0.02, lower_threshold: float = -0.02) -> int:

    if val >= upper_threshold:
        return 1
    elif val <= lower_threshold:
        return -1
    else:
        return 0