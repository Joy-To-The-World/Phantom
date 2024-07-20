import numpy as np
import cv2
import glob

checkerboard_dims = (8, 6) 

objp = np.zeros((8 * 6, 3), np.float32) # figure out what numcorners is
objp[:, :2] = np.mgrid[0:8, 0:6].T.reshape(-1, 2) # ??? idk

objpoints = []
imgpoints = []

images = glob.glob('/home/alicia/bwsi-uav/laboratory_2024/week_1_Hw/camera_calibration_photo_mosaic/calibration photos/*.jpg')
winSize = (11, 11)
zeroZone = (-1, -1)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

for image in images:
    image = cv2.imread(image)
    grayImg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
    # print("In here")
    ret, corners = cv2.findChessboardCorners(grayImg, checkerboard_dims, None) # Find the chessboard corners in the grayscale image

    if ret is True:
        # print("In if statement")
        objpoints.append(objp)
        refCorners = cv2.cornerSubPix(grayImg, corners, winSize, zeroZone, criteria) # Refine corner positions using cornerSubPix
        imgpoints.append(refCorners)
        
        cv2.drawChessboardCorners(image, checkerboard_dims, refCorners, ret)# Draw chessboard corners on the image
        cv2.imshow("Image", image)# Optionally, display the image with drawn corners
        cv2.waitKey(500) # Wait for a short period
    
cv2.destroyAllWindows() # Destroy all OpenCV windows

ret, matrix, distortion, rot, transl = cv2.calibrateCamera(objpoints, imgpoints, grayImg.shape[::-1], None, None) # Calibrate the camera using calibrateCamera with objpoints, imgpoints, and image size
# Get the camera matrix, distortion coefficients, rotation vectors, and translation vectors

np.savez("Calibration_Results", camMatrix = matrix, distCoeffs = distortion, rotVecs = rot, translVecs = transl)# Save the calibration results (camera matrix, distortion coefficients) to a file. 
# A common and convenient format for storing camera calibration data is the NumPy .npz file format,
    # which allows you to store multiple NumPy arrays in a single compressed file.

# Verify the calibration:
mean_error = 0
for i in range(len(objpoints)): # For each pair of object points and image points:
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rot[i], transl[i], matrix, distortion)# Project the object points to image points using projectPoints
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints) # Compute the error between the projected and actual image points
    mean_error += error # Accumulate the error

print("Error: " + str(mean_error/len(objpoints)))