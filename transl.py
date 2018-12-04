import numpy as np

def transl(t_):
    t = np.identity(4)
    t[0, 3] = t_[0]
    t[1, 3] = t_[1]
    t[2, 3] = t_[2]
    return t
