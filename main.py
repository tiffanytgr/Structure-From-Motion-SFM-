import cv2 as cv
import os
import numpy as np

from bundle_adjustment import bundle_adjustment
from plot_utils import viz_3d, viz_3d_matplotlib, draw_epipolar_lines

######################### Path Variables ##################################################
curr_dir_path = os.getcwd()
images_dir = os.path.join(curr_dir_path, 'data/images/observatory')
calibration_file_dir = os.path.join(curr_dir_path, 'data/calibration/observatory')
###########################################################################################
def get_intrinsic_params():
    calib_file_path = os.path.join(calibration_file_dir, 'cameras.txt')
    with open(calib_file_path, 'r') as f:
        lines = f.readlines()
        
        # Select the line containing intrinsic parameters, assuming it's the fourth line (index 3)
        params_line = lines[3]
        
        # Split the line and convert relevant values to floats
        parts = params_line.split()
        
        fx = float(parts[4])   # PARAMS[0]
        fy = float(parts[5])   # PARAMS[1]
        cx = float(parts[6])   # PARAMS[2]
        cy = float(parts[7])   # PARAMS[3]

        # Construct the intrinsic matrix
        K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float32)
    
    return K

# def get_camera_intrinsic_params():
#     # Parse cameras.txt to get intrinsic parameters (K) if needed.
#     K = []
#     with open(os.path.join(calibration_file_dir, 'cameras.txt')) as f:
#         lines = f.readlines()
#         calib_info = [float(val) for val in lines[0].split(' ') if val]
#         row1 = [calib_info[0], calib_info[1], calib_info[2]]
#         row2 = [calib_info[3], calib_info[4], calib_info[5]]
#         row3 = [calib_info[6], calib_info[7], calib_info[8]]

#         K = np.array([row1, row2, row3], dtype=np.float32)
    
#     return K

# # def get_pinhole_intrinsic_params():
# #     calib_file_path = os.path.join(calibration_file_dir, 'cameras.txt')
# #     with open(calib_file_path, 'r') as f:
# #         lines = f.readlines()
        
# #         # Extract intrinsic parameters for cam0 (first line)
# #         cam0_line = lines[0].split('=')[1].strip('[]\n')
        
# #         # Split the line by semicolons to get each row of the matrix
# #         cam0_values = cam0_line.split(';')
        
# #         # Convert each row to a list of floats
# #         row1 = [float(val) for val in cam0_values[0].split()]
# #         row2 = [float(val) for val in cam0_values[1].split()]
# #         row3 = [float(val) for val in cam0_values[2].split()]
        
# #         # Create the intrinsic matrix
# #         K = np.array([row1, row2, row3], dtype=np.float32)
    
# #     return K

def rep_error_fn(opt_variables, points_2d, num_pts):
    P = opt_variables[0:12].reshape(3,4)
    point_3d = opt_variables[12:].reshape((num_pts, 4))

    rep_error = []

    for idx, pt_3d in enumerate(point_3d):
        pt_2d = np.array([points_2d[0][idx], points_2d[1][idx]])

        reprojected_pt = np.matmul(P, pt_3d)
        reprojected_pt /= reprojected_pt[2]

        print("Reprojection Error \n" + str(pt_2d - reprojected_pt[0:2]))
        rep_error.append(pt_2d - reprojected_pt[0:2])

if __name__ == "__main__":
    # Variables 
    iter = 0
    prev_img = None
    prev_kp = None
    prev_desc = None
    K = get_intrinsic_params()
    R_t_0 = np.array([[1,0,0,0], [0,1,0,0], [0,0,1,0]], dtype=np.float32)
    R_t_1 = np.empty((3,4), dtype=np.float32)
    P1 = np.matmul(K, R_t_0)
    P2 = np.empty((3,4), dtype=np.float32)
    pts_4d = []
    X = np.array([])
    Y = np.array([])
    Z = np.array([])

    for filename in os.listdir(images_dir)[0:26]:
        
        file = os.path.join(images_dir, filename)
        img = cv.imread(file, 0)

        resized_img = img
        sift = cv.SIFT_create()
        kp, desc = sift.detectAndCompute(resized_img, None)
        
        if iter == 0:
            prev_img = resized_img
            prev_kp = kp
            prev_desc = desc
        else:
            # FLANN parameters
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=100)
            flann = cv.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(prev_desc, desc, k=2)
            good = []
            pts1 = []
            pts2 = []
            # ratio test as per Lowe's paper
            for i, (m, n) in enumerate(matches):
                if m.distance < 0.7 * n.distance:
                    good.append(m)
                    pts1.append(prev_kp[m.queryIdx].pt)
                    pts2.append(kp[m.trainIdx].pt)
                    
            pts1 = np.array(pts1)
            pts2 = np.array(pts2)
            F, mask = cv.findFundamentalMat(pts1, pts2, cv.FM_RANSAC)
            print("The fundamental matrix \n" + str(F))

            # We select only inlier points
            pts1 = pts1[mask.ravel() == 1]
            pts2 = pts2[mask.ravel() == 1]

            E = np.matmul(np.matmul(np.transpose(K), F), K)
            print("The new essential matrix is \n" + str(E))

            retval, R, t, mask = cv.recoverPose(E, pts1, pts2, K)
            
            R_t_1[:3, :3] = np.matmul(R, R_t_0[:3, :3])
            R_t_1[:3, 3] = R_t_0[:3, 3] + np.matmul(R_t_0[:3, :3], t.ravel())

            P2 = np.matmul(K, R_t_1)

            pts1 = np.transpose(pts1)
            pts2 = np.transpose(pts2)

            points_3d = cv.triangulatePoints(P1, P2, pts1, pts2)
            points_3d /= points_3d[3]

            opt_variables = np.hstack((P2.ravel(), points_3d.ravel(order="F")))
            num_points = len(pts2[0])
            rep_error_fn(opt_variables, pts2, num_points)

            X = np.concatenate((X, points_3d[0]))
            Y = np.concatenate((Y, points_3d[1]))
            Z = np.concatenate((Z, points_3d[2]))

            R_t_0 = np.copy(R_t_1)
            P1 = np.copy(P2)
            prev_img = resized_img
            prev_kp = kp
            prev_desc = desc

        iter += 1

    pts_4d.append(X)
    pts_4d.append(Y)
    pts_4d.append(Z)

    viz_3d(np.array(pts_4d))
