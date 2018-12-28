from skew import *
from quat2rot import *
from rot2quat import *
from transl import *

def handEye(bHg, wHc, start_idx=None, end_idx=None):

    if start_idx is None:
        start_idx = 0
    if end_idx is None:
        end_idx = bHg.shape[2]
    bHg = bHg[:,:,start_idx:end_idx]
    wHc = wHc[:,:,start_idx:end_idx]

    M = bHg.shape[2]            # number of camera positions
    K = int((M * M - M) / 2)    # number of camera position pairs

    A = np.zeros((3 * K, 3))
    B = np.zeros((3 * K, 1))

    Hg = bHg
    Hc = np.zeros((4, 4, M))

    for i in range(M):
        Hc[:,:,i] = np.linalg.inv(wHc[:,:,i])

    k = 0
    for i in range(M):
        for j in range(i+1, M):
            Hgij = np.matmul(np.linalg.inv(Hg[:,:,j]), Hg[:,:,i])
            Pgij = 2 * rot2quat(Hgij)

            Hcij = np.matmul(Hc[:,:,j], np.linalg.inv(Hc[:,:,i]))
            Pcij = 2 * rot2quat(Hcij)

            A[3*k:3*(k+1)] = skew(Pgij + Pcij)
            B[3*k:3*(k+1),0] = Pcij - Pgij
            k += 1


    Pcg_, res1 = np.linalg.lstsq(A, B)[0:2]

    res1 = res1[0] / A.shape[0]

    Pcg = 2 * Pcg_ / np.sqrt(1 + np.matmul(np.transpose(Pcg_), Pcg_))

    Rcg = quat2rot(Pcg / 2)

    # Calculate translational component
    k = 0
    for i in range(M):
        for j in range(i+1, M):
            Hgij = np.matmul(np.linalg.inv(Hg[:,:,j]), Hg[:,:,i])
            Hcij = np.matmul(Hc[:,:,j], np.linalg.inv(Hc[:,:,i]))

            A[3*k:3*(k+1)] = Hgij[:3,:3] - np.identity(3)
            B[3*k:3*(k+1), 0] = np.matmul(Rcg[:3,:3], Hcij[:3,3]) - Hgij[:3,3]
            k += 1

    Tcg, res2 = np.linalg.lstsq(A, B)[0:2]

    res2 = res2[0] / A.shape[0]

    print("Residuals:")
    print(res1)
    print(res2)

    gHc = np.matmul(transl(Tcg.squeeze()), Rcg)

    return gHc
