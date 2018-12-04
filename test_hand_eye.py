from robot_kinematics import *
from hand_eye import *

if __name__ == '__main__':
    M = 3
    # 1. randomly generate transforms from gripper to base
    bHg_list = []
    for m in range(M):
        phi_angle_ary = [random.uniform(-160, 160),
                         random.uniform(-90, 0),
                         random.uniform(-90, 0),
                         random.uniform(-90, 90),
                         random.uniform(-90, 90),
                         random.uniform(-180, 180)]

        T_list = g2b(phi_angle_ary)

        H = np.identity(4)
        for T in T_list:
            H = np.matmul(T, H)
        bHg_list.append(np.linalg.inv(H))

    bHg = np.array(bHg_list)
    bHg = np.transpose(bHg, (1, 2, 0))

    # for debug
    bHg = np.array([[[ -1.00000000e+00,  -9.65925826e-01,  -1.00000000e+00],
        [  3.74939946e-33,  -1.58480958e-17,   1.15631411e-17],
        [ -1.22464680e-16,   2.58819045e-01,  -1.32167307e-16],
        [  4.83000000e+02,   4.42624229e+02,   4.83000000e+02]],

       [[ -7.49879891e-33,   1.58480958e-17,   1.15631411e-17],
        [ -1.00000000e+00,  -1.00000000e+00,  -9.84807753e-01],
        [  6.88753506e-49,  -2.08644139e-18,  -1.73648178e-01],
        [ -2.00000000e+01,  -2.00000000e+01,   7.08911572e+00]],

       [[ -1.22464680e-16,   2.58819045e-01,  -1.32167307e-16],
        [ -4.59169004e-49,   2.08644139e-18,  -1.73648178e-01],
        [  1.00000000e+00,   9.65925826e-01,   9.84807753e-01],
        [  4.29000000e+02,   4.34315571e+02,   4.31369991e+02]],

       [[  0.00000000e+00,   0.00000000e+00,   0.00000000e+00],
        [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00],
        [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00],
        [  1.00000000e+00,   1.00000000e+00,   1.00000000e+00]]])

    # 2. randomly generate the gHc, the pose of camera relative to the gripper
    random_angle_ary = [random.uniform(-160, 160),
                        random.uniform(-90, 0),
                        random.uniform(-90, 0),
                        random.uniform(-90, 90),
                        random.uniform(-90, 90),
                        random.uniform(-180, 180)]

    T_list = g2b(random_angle_ary)
    gHc = np.identity(4)
    for T in T_list:
        gHc = np.matmul(T, gHc)

    gHc = np.array([[0, 1, 0, 0],
                    [1, 0, 0, 0],
                    [0, 0, -1, 0],
                    [0, 0, 0, 1]])

    '''
    gHc = np.array([[  6.15885419e-02,   9.97857037e-01,   2.20949347e-02,
         -7.21162363e+01],
       [  9.96495316e-01,  -6.02189465e-02,  -5.80582830e-02,
         -1.78931558e+00],
       [ -5.66033325e-02,   2.55932239e-02,  -9.98068660e-01,
         -5.43625544e+00],
       [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          1.00000000e+00]])
    '''

    # 3. randomly generate wHb, the transform between the robot base and the world frame
    random_angle_ary = [random.uniform(-160, 160),
                        random.uniform(-90, 0),
                        random.uniform(-90, 0),
                        random.uniform(-90, 90),
                        random.uniform(-90, 90),
                        random.uniform(-180, 180)]

    T_list = g2b(random_angle_ary)
    wHb = np.identity(4)
    for T in T_list:
        wHb = np.matmul(T, wHb)

    wHb = np.identity(4)

    '''
    wHb = np.array([[  2.74548324e-02,   9.99304660e-01,   2.52275054e-02,
          5.85541375e+01],
       [ -9.99620676e-01,   2.73910793e-02,   2.86928698e-03,
          5.29233569e+02],
       [  2.17628320e-03,  -2.52967118e-02,   9.99677618e-01,
          4.47623715e+00],
       [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          1.00000000e+00]])
    '''

    # 4. calculate wHc, pose of camera relatively to the world
    #   wHc = gHc * bHg * wHb
    wHc_list = []
    for m in range(M):
        # wHc_list.append(gHc.dot(bHg[:,:,m]).dot(wHb))
        wHc_list.append(wHb.dot(bHg[:,:,m]).dot(gHc))

    wHc = np.array(wHc_list)
    wHc = np.transpose(wHc, (1, 2, 0))

    '''
    wHc = np.array([[[ -9.98921277e-01,  -9.98753393e-01,  -9.78060977e-01,
                       9.96422786e-01],
                    [  3.34267294e-02,   4.04017027e-02,   2.85543375e-02,
                      -4.41003679e-02],
                    [  3.22325175e-02,   2.93148783e-02,   2.06352549e-01,
                      -7.20887568e-02],
                    [  6.62821956e+01,   6.44596519e+01,   9.39488928e+01,
                       6.21490966e+01]],

                   [[  3.41076865e-02,   4.70410852e-02,   3.44335752e-02,
                      -4.76405306e-02],
                    [  9.99201422e-01,   9.58255525e-01,   9.99095367e-01,
                      -9.97703374e-01],
                    [  2.08130872e-02,   2.82016818e-01,   2.49554948e-02,
                      -4.81493161e-02],
                    [ -2.49555829e+01,   1.92633237e+01,  -2.44798893e+01,
                       1.10128316e+02]],

                   [[ -3.15110639e-02,  -1.66971845e-02,  -2.05453288e-01,
                      -6.97997934e-02],
                    [  2.18900123e-02,   2.83044258e-01,   3.15134517e-02,
                       5.14114223e-02],
                    [ -9.99263669e-01,  -9.58961497e-01,  -9.78159419e-01,
                      -9.96235341e-01],
                    [  4.29572193e+02,   4.16435951e+02,   4.31718974e+02,
                       4.27350402e+02]],

                   [[  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
                       0.00000000e+00],
                    [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
                       0.00000000e+00],
                    [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
                       0.00000000e+00],
                    [  1.00000000e+00,   1.00000000e+00,   1.00000000e+00,
                       1.00000000e+00]]])
    '''

    gHc_ = handEye(bHg, wHc)

    print(gHc_)

    # print(np.sum(gHc - gHc_))

    # import pdb
    # pdb.set_trace()
