import cv2
import os
import pickle
import numpy as np
from numpy.linalg import inv
import glob
import matplotlib.pyplot as plt
import pdb

def get_inv_order(order):
    leng = np.shape(order)[-1]
    one_hot = np.eye(leng)[order]
    one_hot_inv = inv(one_hot)
    inv_order = np.argmax(one_hot_inv, 1)
    return inv_order

def change_order(arr, h_dst, w_dst, order_0, order_1):
    if order_0 == 1:
        arr = np.reshape(np.transpose(np.reshape(arr, [w_dst, h_dst, -1]), [1, 0, 2]), [h_dst * w_dst, -1])

    if order_1 == 1:
        arr = np.reshape(np.reshape(arr, [h_dst, w_dst, -1])[:,::-1], [h_dst * w_dst, -1])
    if order_1 == 2:
        arr = np.reshape(np.reshape(arr[::-1], [h_dst, w_dst, -1])[:,::-1], [h_dst * w_dst, -1])
    if order_1 == 3:
        arr = arr[::-1]
    return arr

def calibrate(img_dir, flip=False, show_img=False, img_format='jpg', save_dir=None):

    img_name_list = os.listdir(img_dir)
    img_idx_list = [int(e.split('.')[0]) for e in img_name_list]
    img_num = np.max(img_idx_list) + 1

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    w = 7
    h = 9
    homo_k = 4
    homo_size = (1440, 1080)

    grid_width = 23.8
    grid_height = 23.75

    objp = np.zeros((w * h, 3)).astype(np.float32)
    for i in range(w * h):
        objp[i, 0] = i // w * grid_width
        objp[i, 1] = i % w * grid_height
        objp[i, 2] = 0.0

    objpoints = [] # points3D in world
    imgpoints = [] # points2D in img
    imgpoints_ori = [] # points2D in img
    order = []     # raw corner order by cv2

    corner4 = np.array((objp[0], objp[6], objp[56], objp[62]))
    pts_dst_uo = homo_k * corner4[:,:2] + (np.array(homo_size) - homo_k * objp[62][:2]) / 2

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

            corners_sq = np.squeeze(corners)

            # determine order_0
            d_70 = (corners_sq[7][0] - corners_sq[0][0]) ** 2 + (corners_sq[7][1] - corners_sq[0][1]) ** 2
            d_76 = (corners_sq[7][0] - corners_sq[6][0]) ** 2 + (corners_sq[7][1] - corners_sq[6][1]) ** 2
            order_0 = int(d_70 > d_76)
            
            # order the pts_dst according to pts_src
            if order_0:
                pts_src = np.array((corners_sq[0], corners_sq[8], corners_sq[54], corners_sq[62]))
                sort = np.array([0, 2, 1, 3])
            else:
                pts_src = np.array((corners_sq[0], corners_sq[6], corners_sq[56], corners_sq[62]))
                sort = np.array([0, 1, 2, 3])

            pts_3d_src = np.concatenate((pts_src, np.ones([4, 1])), axis = 1)
            sort = sort.astype(int)
            sort_inv = get_inv_order(sort)
            
            pts_dst = pts_dst_uo[sort_inv]
            pts_3d_dst = np.concatenate((pts_dst, np.ones([4, 1])), axis = 1)
            
            # Homography 
            h_s2d, status = cv2.findHomography(pts_src, pts_dst)
            img_homo = cv2.warpPerspective(img, h_s2d, (1440, 1080))

            # Circle detection
            # gray_homo = cv2.cvtColor(img_homo, cv2.COLOR_BGR2GRAY)
            circles = cv2.HoughCircles(img_homo, cv2.HOUGH_GRADIENT, 1, 100, 
                    param1 = 80, param2 = 30, minRadius = 30, maxRadius = 90)
            circle = circles[0][0]
            img_circle = cv2.circle(img_homo, (int(circle[0]), int(circle[1])), int(circle[2]), (0, 0, 255), 1, 8, 0)

            # determine order_1
            d_c4 = (pts_dst[:,0] - circle[0]) ** 2 + (pts_dst[:,1] - circle[1]) ** 2
            order_1 = np.argmin(d_c4)
            order.append([order_0, order_1])
           
            # change corners order
            corners_order = change_order(corners_sq, h, w, order_0, order_1)[:, np.newaxis, :]
            imgpoints.append(corners_order)

            # test corners order
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            if show_img:
                cv2.drawChessboardCorners(img_rgb, (w, h), corners_order, ret)
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

    print('total error: ', total_error / len(objpoints))
    print('mtx:\n', mtx)
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
