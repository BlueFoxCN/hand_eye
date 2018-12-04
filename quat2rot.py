import numpy as np
from skew import *

def quat2rot(q):
    p = np.matmul(np.transpose(q), q).squeeze()
    if p > 1:
        print('Warning: quat2rot: quaternion greater than 1')
    w = np.sqrt(1 - p)
    R = np.identity(4)
    R[:3, :3] = 2 * np.matmul(q, np.transpose(q)) + 2 * w * skew(q) + np.identity(3) - 2 * np.diag((p, p, p))
    return R

