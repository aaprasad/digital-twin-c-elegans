import numpy as np


def clock_position(distance):
    pos_base = [
        (0, distance),
        (distance * np.sin(np.pi / 6), distance * np.cos(np.pi / 6)),
        (distance * np.sin(np.pi / 3), distance * np.cos(np.pi / 3))
    ]
    pos = []
    for x, y in pos_base:
        pos.append((x, y))
        pos.append((y, -x))
        pos.append((-x, -y))
        pos.append((-y, x))
    return pos
