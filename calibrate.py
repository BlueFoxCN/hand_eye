import cv2
import os
import pickle
import numpy as np
from numpy.linalg import inv
import glob
import matplotlib.pyplot as plt
import pdb

def calibrate(img_dir, flip=False, show_img=False, img_format='jpg', save_dir=None):

    img_name_list = os.listdir(img_dir)
    img_idx_list = [int(e.split('.')[0]) for e in img_name_list]
    img_num = np.max(img_idx_list) + 1

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    w = 6
    h = 9

    grid_width = 22.65
    grid_height = 22.65
    # grid_width = 23.8
    # grid_height = 23.75

    objp = np.zeros((w * h, 3)).astype(np.float32)
    for i in range(w * h):
        objp[i, 0] = i // w * grid_width
        objp[i, 1] = i % w * grid_height
        objp[i, 2] = 0.0

    objpoints = [] # points3D in world
    imgpoints = [] # points2D in img
    imgpoints_ori = [] # points2D in img
    order = []     # raw corner order by cv2
    idxes = []

    for i in range(img_num):
        fname = os.path.join(img_dir, "%d.%s" % (i, img_format))
        img = cv2.imread(fname)
        if flip == True:
            img = cv2.flip(img, 0)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = img.copy()

        # find corners
        ret, corners = cv2.findChessboardCorners(img, (w,h), None)

        if ret:
            idxes.append(i)
            objpoints.append(objp)
            cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            imgpoints.append(corners)

            # test corners order
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            if show_img:
                cv2.drawChessboardCorners(img_rgb, (w, h), corners, ret)
                cv2.imshow('order', img_rgb)
                cv2.waitKey(0)

    # calibrate
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, 
                                                       imgpoints, 
                                                       gray.shape[::-1], 
                                                       None, 
                                                       None)

    '''
    ret: retval re-projection error
    mtx: camera matrix
    dist: distort coefficients
    rvecs: extrinsic rotation parameters
    tvecs: extrinsic translation parameters
    '''

    ''' show distortion result
    for i in range(img_num):
        fname = os.path.join(img_dir, "%d.%s" % (i, img_format))
        img = cv2.imread(fname)
        h,  w = img.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

        # undistort
        dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
        cv2.imshow('distort', dst)
        cv2.waitKey(0)
    '''


    # re-project error
    total_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        total_error += error

    print('Camera calibration error: ', total_error / len(objpoints))
    print('Camera matrix:\n', mtx)
    wHc_dict = {}
    for i, idx in enumerate(idxes):
        rvec = rvecs[i]
        tvec = tvecs[i]

        A = cv2.Rodrigues(rvec)[0]
        B = tvec.reshape((3, 1))
        C = np.zeros((1, 3))
        D = np.ones((1, 1))
        cHw = np.vstack((np.hstack((A, B)),
                         np.hstack((C, D))
                       ))
        wHc = np.linalg.inv(cHw)
        wHc_dict[idx] = wHc

        # print('wHc:\n', wHc)
    
        if save_dir != None:
            np.savez("%s/%d.npz" % (save_dir, idx), wHc=wHc)


    return wHc_dict

if __name__ == '__main__':

    img_dir = 'cali_imgs'
    save_dir = 'result_2'

    calibrate(img_dir, save_dir=save_dir)
