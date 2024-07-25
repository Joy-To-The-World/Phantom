import cv2
import numpy as np
import glob

# Load Calibration Parameters:
#    Function load_calibration(calibration_file): calibration_matrix
#   Load calibration data from the file
#         Extract camera matrix and distortion coefficients
#         Return camera matrix and distortion coefficients

def load_calibration(calibration_file):
    calibration_data = np.load(calibration_file)
    mtx = calibration_data['arr_0']  
    dist = calibration_data['arr_1'] 
    return mtx, dist

# calibration_file = 'calibration_matrix.npz'
# mtx, dist = load_calibration(calibration_file)

# print("Camera Matrix:\n", mtx)
# print("Distortion Coefficients:\n", dist)


# Undistort Image:
#     Function undistort_image(image, camera_matrix, dist_coeffs):
#         Get image dimensions (height, width)
#         Compute new camera matrix for undistortion
#         Undistort the image (use cv2 undistort)
#         Crop the undistorted image using ROI
#         Return undistorted image

def undistort_image(image, camera_matrix, dist_coeffs):
    h, w = image.shape[:2]
    new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w,h), 1, (w,h))
    undistorted_image = cv2.undistort(image, camera_matrix, dist_coeffs, None, new_camera_mtx)

    x, y, w, h = roi
    undistorted_image = undistorted_image[y:y+h, x:x+w]

    return undistorted_image, roi


# Harris Corner Detection:
#     Function harris_corner_detection(image):
#         Convert the image to grayscale
#         Apply Harris corner detection
#         Dilate corners
#         Mark corners on the image
#         Return image with marked corners and detected corners

def harris_corner_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)

    dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    dst = cv2.dilate(dst, None)

    image[dst > 0.01 * dst.max()] = [0, 0, 255] #marks up the image

    return image, dst
#dst is a 2D array


# Match Features Between Images:
#     Function match_features(image1, image2):
#         Detect keypoints and descriptors in image1 using SIFT
#         Detect keypoints and descriptors in image2 using SIFT
#         Match descriptors using brute-force matcher
#         Extract matched points from both images
#         Return matched points from image1 and image2

def match_features(image1, image2):
    # Example: Match features using SIFT
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(image1, None)
    kp2, des2 = sift.detectAndCompute(image2, None)
    
    # Match descriptors using brute-force matcher
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    
    # Apply ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    
    # Ensure enough matches for homography calculation
    if len(good_matches) < 4:
        return None, None
    
    points1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    points2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    #this part extracts points that fit the criteria
    #retrieves the coordinates of the keypoint in the first 
    # image (image1) corresponding to the descriptor index m.queryIdx
    print(points1, points2)
    
    return points1, points2

# Create Mosaic:
#     Function create_mosaic(images, camera_matrix, dist_coeffs):
#         Undistort all images using undistort_image function
#         Initialize mosaic with the first undistorted image
#         For each subsequent undistorted image:
#             Detect Harris corners in both mosaic and current image using harris_corner_detection
#             Match features between mosaic and current image using match_features
#             Estimate homography using matched points
#             Warp mosaic image using the estimated homography
#             Blend current image into mosaic
#         Return final mosaic image

def create_mosaic(images, camera_matrix, dist_coeffs):
    undistorted_images = []
    rois = []
    
    for img in images:
        undistorted_img, roi = undistort_image(img, camera_matrix, dist_coeffs)
        undistorted_images.append(undistorted_img)
        rois.append(roi)

    mosaic = undistorted_images[0].copy()

    for i in range(1, len(undistorted_images)):
        mosaic_corners, _ = harris_corner_detection(mosaic)
        current_corners, _ = harris_corner_detection(undistorted_images[i])

        points1, points2 = match_features(mosaic_corners, current_corners)

        if points1 is None or points2 is None:
            continue

        points1 = points1.reshape(-1, 1, 2)
        points2 = points2.reshape(-1, 1, 2)

        H, _ = cv2.findHomography(points1, points2, cv2.RANSAC, 5.0)

        warped_image = cv2.warpPerspective(undistorted_images[i], H, (mosaic.shape[1], mosaic.shape[0]))

        mask = np.zeros_like(mosaic, dtype=np.uint8)
        mask[warped_image[:, :, 0] != 0] = 255

        # Ensure mask is grayscale (single-channel)
        mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        # Perform seamlessClone with the correct mask and ROI
        roi = rois[i]
        if roi[0] >= 0 and roi[1] >= 0 and (roi[0] + roi[2]) <= mosaic.shape[1] and (roi[1] + roi[3]) <= mosaic.shape[0] \
            and roi[0] >= 0 and roi[1] >= 0 and (roi[0] + roi[2]) <= warped_image.shape[1] and (roi[1] + roi[3]) <= warped_image.shape[0]:
            try:
                blend = cv2.seamlessClone(warped_image, mosaic, mask_gray, (0, 0), cv2.NORMAL_CLONE)
            except cv2.error as e:
                print(f"Error in seamlessClone: {e}")
                print(f"Dimensions: mosaic={mosaic.shape}, warped_image={warped_image.shape}, mask={mask_gray.shape}")
                continue

            mosaic = blend

    return mosaic




# Main:
#     Load camera matrix and distortion coefficients from calibration file
#     Load images from specified directory
#     Create mosaic using create_mosaic function
#     Save the mosaic image to a file


def main():
    calibration_file = "calibration_matrix.npz"
    camera_matrix, dist_coeffs = load_calibration(calibration_file)
    images_path = 'International Village - 15 Percent Overlap/*.jpg'
    images_files = glob.glob(images_path)
    images = [cv2.imread(img) for img in images_files]

    mosaic_image = create_mosaic(images, camera_matrix, dist_coeffs)

    cv2.imshow('Mosaic', mosaic_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save the mosaic image
    cv2.imwrite('mosaic_image.jpg', mosaic_image)

if __name__ == "__main__":
    main()