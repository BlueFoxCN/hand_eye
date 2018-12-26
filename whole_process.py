from robot_kinematics import *
from hand_eye import *
from calibrate import *

data_dir = "ir_imgs"

# 1. get the gripper poses
fname = '%s/calibrationValueConfig.txt' % data_dir
f = open(fname, 'r')
lines = f.readlines()
bHg_list = []
for line in lines:
	angles = [float(e.strip()) for e in line.split(',')]
	T_list = g2b(angles)
	T = np.identity(4)
	for t in T_list:
		T = np.matmul(t, T)
	bHg_list.append(T)

# 2. calibrate camera and get extrinsic matrix
wHc_dict = calibrate("%s/mannual" % data_dir, flip=False, show_img=False, img_format='bmp')

# 3. filter bHg and wHc
wHc = []
bHg = []
for k in wHc_dict:
	bHg.append(bHg_list[k])
	wHc.append(wHc_dict[k])

# 4. do hand-eye calibration
bHg = np.array(bHg)
bHg = np.transpose(bHg, (1, 2, 0))
wHc = np.array(wHc)
wHc = np.transpose(wHc, (1, 2, 0))

gHc = handEye(bHg[:,:,:], wHc[:,:,:])

print(gHc[:3,:3])
print(gHc[:,3])

# import pdb
# pdb.set_trace()
