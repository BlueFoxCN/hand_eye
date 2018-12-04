import numpy as np

def skew(v):
    v = np.squeeze(v)
    s = np.zeros((3, 3))
    s[0, 1] = -v[2]
    s[0, 2] = v[1]
    s[1, 0] = v[2]
    s[1, 2] = -v[0]
    s[2, 0] = -v[1]
    s[2, 1] = v[0]
    return s
