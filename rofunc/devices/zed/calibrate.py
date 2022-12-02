import os, csv, sys
from utils import *
import cv2 as cv
# from matplotlib import pyplot as plt
import numpy as np


# build the transformation matrix given 1 image calibration ; return the inverse of the matrix and the matrix
def calc_transfer_matrix(img_path, camera_matrix, dist_coeff, criteria, objp):
    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    objpoints.append(objp)

    imgpoints = []  # 2d points in image plane.

    img_calibration = cv.cvtColor(cv.imread(img_path, 1), cv.COLOR_BGR2RGB)
    gray_img = cv.cvtColor(img_calibration, cv.COLOR_RGB2GRAY)

    # axis along 3 squares
    axis = np.float32([[0.07, 0, 0], [0, 0.07, 0], [0, 0, 0.07]]).reshape(-1, 3)

    # find pattern in chess board (find the inside chess board corners)
    ret, corners = cv.findChessboardCorners(gray_img, (7, 4), None)

    # If found, add object points, image points (after refining them)
    if ret == True:

        corners2 = cv.cornerSubPix(gray_img, corners, (3, 3), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Final rotation and translation vectors using cv2 intrinsics
        _, rvecs, tvecs = cv.solvePnP(objp, corners2, camera_matrix, dist_coeff)

        img_calibration = cv.drawChessboardCorners(img_calibration, (4, 7), corners2, ret)

        # PLOTS THE CALIBRATION STUFF ON THE IMAGE
        # imgpts, jac = cv.projectPoints(axis, rvecs, tvecs, camera_matrix, dist_coeff)

        # img_calibration = draw(img_calibration, corners2, imgpts)

        # plt.figure()
        # plt.imshow(img_calibration, cmap='gray')
        # plt.title('pose estimation')
        # plt.show()

        # rotation matrix
        rotation = np.zeros((3, 3))
        cv.Rodrigues(rvecs,
                     rotation)  # https://docs.opencv.org/trunk/d9/d0c/group__calib3d.html#ga61585db663d9da06b68e70cfbf6a1eac

        # transition matrix [[rotation tvecs]/[0 0 0 1]] from the model coordinate system to the camera coordinate system
        trans_mat = np.zeros((4, 4))
        trans_mat[0:3, 0:3] = rotation
        trans_mat[0:3, 3] = tvecs.flatten()
        trans_mat[3, 3] = 1

        return trans_mat
    else:
        raise Exception("Could not locate all corners of chessboard....")
        # raise Exception("Could not find corners of chessboard.... Exiting")


def get_transfer_matrix(path_intrinsics, img_path, criteria, objp):
    camera_matrix = np.load(os.path.join(path_intrinsics, 'camera_matrix.npy'))
    trans_depth2color = np.load(os.path.join(path_intrinsics, 'trans_depth2color.npy'))
    dist_coeff = np.load(os.path.join(path_intrinsics, 'dist_coeff.npy'))

    # transformation matrices
    trans_world2color = calc_transfer_matrix(img_path, camera_matrix, dist_coeff, criteria, objp)

    print("Transfer Matrix bag %s= \n %s" % (os.path.basename(path_intrinsics), trans_world2color))

    return trans_world2color, trans_depth2color, camera_matrix, dist_coeff


def get_calibration_data(path_intrinsics, img_path):
    # Chessboard settings
    number_of_squars = 7
    width = 0.042

    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 300, 0.0001)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) .... (7,4,0)
    objp = np.zeros((4 * number_of_squars, 3), np.float32)
    objp[:, :2] = np.mgrid[0:number_of_squars, 0:4].T.reshape(-1,
                                                              2) * width  # the size of a square is 7cm ; it is in meters

    trans_world2color, trans_depth2color, camera_matrix, dist_coeff = get_transfer_matrix(path_intrinsics, img_path,
                                                                                          criteria, objp)
    trans_color2world = np.linalg.inv(trans_world2color)

    # SAVE FILES
    name_trans_world2color = os.path.join(path_intrinsics, "trans_world2color.npy")
    name_trans_color2world = os.path.join(path_intrinsics, "trans_color2world.npy")

    np.save(name_trans_world2color, trans_world2color)
    np.save(name_trans_color2world, trans_color2world)


if __name__ == "__main__":

    import shutil

    #  # #VOLANS
    # right_cam = '20190122_113806'
    # right_calibration_img = '008530.jpg'

    # left_cam = '20190122_113808'
    # left_calibration_img = '008559.jpg'

    # computer_run_folder = 'volans/calibration2'
    # which_chess_board = 'Calibration_tiny_chessboard'

    # ASUS
    # right_cam = '20190121_181659'
    # right_calibration_img = '104474.jpg'

    # left_cam = '20190121_181702'
    # left_calibration_img = '041308.jpg'

    # computer_run_folder = 'asus/calibration2'
    # which_chess_board = 'Calibration_tiny_chessboard'

    # date_folder = '2019_22_01_Acq'

    cam1_calibration_img = "cam1.jpg"
    cam2_calibration_img = "cam2.jpg"
    cam3_calibration_img = "cam3.jpg"
    cam4_calibration_img = "cam4.jpg"

    root = os.getcwd()

    cam_list = ['cam1', 'cam2', 'cam3', 'cam4']

    list_path_img_calibration = [os.path.join(root, "Data", cam_list[0], "intrinsics", cam1_calibration_img),
                                 os.path.join(root, "Data", cam_list[1], "intrinsics", cam2_calibration_img),
                                 os.path.join(root, "Data", cam_list[2], "intrinsics", cam3_calibration_img),
                                 os.path.join(root, "Data", cam_list[3], "intrinsics", cam4_calibration_img)]

    list_path_intrinsics = [os.path.join(root, "Data", cam_list[0], "intrinsics"),
                            os.path.join(root, "Data", cam_list[1], "intrinsics"),
                            os.path.join(root, "Data", cam_list[2], "intrinsics"),
                            os.path.join(root, "Data", cam_list[3], "intrinsics")]

    ######################################################
    ####################### CHECKS #######################
    ######################################################

    for path in list_path_intrinsics:
        if not os.path.isfile(os.path.join(path, 'camera_matrix.npy')):
            raise Exception("ERROR: camera_matrix.npy not found...\n Path given:%s" % (path))
        if not os.path.isfile(os.path.join(path, 'dist_coeff.npy')):
            raise Exception("ERROR: dist_coeff.npy not found...\n Path given:%s" % (path))
        if not os.path.isfile(os.path.join(path, 'trans_depth2color.npy')):
            raise Exception("ERROR: trans_depth2color.npy not found...\n Path given:%s" % (path))
    for img in list_path_img_calibration:
        if not os.path.isfile(img):
            raise Exception("ERROR: calibration image not found...\nImage path:%s" % (img))

    for i in range(len(list_path_intrinsics)):
        get_calibration_data(list_path_intrinsics[i], list_path_img_calibration[i])