import numpy as np
import cv2 as cv
import glob
 
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
 
images = glob.glob('laboratory_2024/week_1_Hw/camera_calibration_photo_mosaic/calibration_photos/*.jpg')
# print(images)
 
chessboard_dims=(8,6)
corners=chessboard_dims[0]*chessboard_dims[1]
 
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((corners,3), np.float32)
objp[:,:2] = np.mgrid[0:chessboard_dims[0],0:chessboard_dims[1]].T.reshape(-1,2)
# print(objp)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
# print(images)

for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    #print(img)
    #print(gray)

    # Find the chess board corners
    ret, corns = cv.findChessboardCorners(gray, chessboard_dims, None)
    # print(ret)
    # print(corners)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray,corns, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)
 
    # Draw and display the corners
    # cv.drawChessboardCorners(img, chessboard_dims, corners2, ret)
    # cv.imshow('img', img)
    # cv.waitKey(500)
 
# cv.destroyAllWindows()

# print(objpoints)
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
np.savez("camera_matrix.npz", mtx, dist)
# print(len(rvecs))
# print(len(tvecs))
# print(corners)
# print(mtx)
mean_error=0
for i in range(len(images)):
    objpt=objpoints[0][i].astype(np.float32)
    imgpt=imgpoints[0][i].astype(np.float32)
    # print(imgpt)
    # print(objpt)
    # print(cv.projectPoints(objpt, rvecs[i], tvecs[i], mtx, dist))
    projectedobjpt, _=cv.projectPoints(objpt, rvecs[i], tvecs[i], mtx, dist)
    projectedobjpt=projectedobjpt.astype(np.float32)
    # print(projectedobjpt)
    pterr=0
    pterr = np.sqrt((imgpt[0][0]-projectedobjpt[0][0][0])**2+(imgpt[0][0]-projectedobjpt[0][0][1])**2)
    # print(pterr)
    mean_error += pterr
print(mean_error/len(objpoints))