import numpy as np
import cv2

def rot2quat(R):
    w4 = 2 * np.sqrt(1 + np.trace(R[:3, :3]))
    q = [(R[2,1] - R[1,2]) / w4,
         (R[0,2] - R[2,0]) / w4,
         (R[1,0] - R[0,1]) / w4]
    return np.array(q)

def rot2quat_plus(R):
    rvec = cv2.Rodrigues(R[:3,:3])[0]
    theta = np.linalg.norm(rvec)
    norm_rvec = rvec / theta
    q = np.sin(theta / 2) * norm_rvec
    return np.squeeze(q)

'''
def rot2quat_plus(R):

    s_list = [R[0,0] + R[1,1] + R[2,2],
              R[0,0] - R[1,1] - R[2,2],
              R[1,1] - R[0,0] - R[2,2],
              R[2,2] - R[0,0] - R[1,1]]

    idx = np.argmax(s_list)

    if idx == 0:
        s = np.sqrt(1 + R[0,0] + R[1,1] + R[2,2]) * 2
        q = [(R[2,1] - R[1,2]) / s,
             (R[0,2] - R[2,0]) / s,
             (R[1,0] - R[0,1]) / s]
    elif idx == 1:
        s = np.sqrt(1 + R[0,0] - R[1,1] - R[2,2]) * 2
        q = [0.25 * s,
             (R[0,1] + R[1,0]) / s,
             (R[0,2] + R[2,0]) / s]
    elif idx == 2:
        s = np.sqrt(1 + R[1,1] - R[0,0] - R[2,2]) * 2
        q = [(R[0,1] + R[1,0]) / s,
             0.25 * s,
             (R[1,2] + R[2,1]) / s]
    else:
        s = np.sqrt(1 + R[2,2] - R[0,0] - R[1,1]) * 2
        q = [(R[0,2] + R[2,0]) / s,
             (R[1,2] + R[2,1]) / s,
             0.25 * s]

    return np.array(q)
'''
