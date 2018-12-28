import argparse

from robot_kinematics import *
from hand_eye import *
from calibrate import *

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', required=True)
parser.add_argument('--img_sub_dir', required=True)
parser.add_argument('--img_format', default='bmp')
parser.add_argument('--start_idx', type=int)
parser.add_argument('--end_idx', type=int)
args = parser.parse_args()

# 1. get the gripper poses
fname = '%s/calibrationValueConfig.txt' % args.data_dir
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
wHc_dict = calibrate("%s/%s" % (args.data_dir, args.img_sub_dir), show_img=False, img_format=args.img_format)

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

gHc = handEye(bHg, wHc, start_idx=args.start_idx, end_idx=args.end_idx)


print("Hand-eye calibration result:")

for idx in range(4):
    line_content = '[' + ', '.join([str(e) for e in gHc[idx]]) + ']'
    if idx == 0:
        line_content = 'np.array([' + line_content + ','
    elif idx == 3:
        line_content += '])'
    else:
        line_content += ','
    print(line_content)
    

