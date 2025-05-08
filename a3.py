import cv2
import numpy as np
from matplotlib import pyplot as plt

def compute_ego_motion_and_depth(image1_path, image2_path, camera_matrix):
    # Load images
    img1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)

    plt.subplot(1,3,1)
    plt.imshow(img1, cmap="gray")
    plt.title("Image 1")
    plt.subplot(1,3,2)
    plt.imshow(img2, cmap="gray")
    plt.title("Image 2")
    #plt.show()

    # Detect ORB keypoints and descriptors
    orb = cv2.ORB_create()
    keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(img2, None)

    # Convert descriptors to CV_32F for FLANN matcher
    descriptors1 = np.float32(descriptors1)
    descriptors2 = np.float32(descriptors2)

    # Match features using FLANN matcher
    index_params = dict(algorithm=1, trees=5)  # FLANN parameters for KDTree
    search_params = dict(checks=50)  # Number of checks for nearest neighbor search
    matcher = cv2.FlannBasedMatcher(index_params, search_params)
    
    matches = matcher.knnMatch(descriptors1, descriptors2, k=2)

    # Filter matches using Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # Extract matched keypoints
    pts1 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches])
    pts2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches])

    # Compute Essential matrix
    E, mask = cv2.findEssentialMat(pts1, pts2, camera_matrix, method=cv2.RANSAC, prob=0.999, threshold=1.0)

    # Recover pose (rotation and translation)
    _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, camera_matrix)

    print("Rotation Matrix:\n", R)
    print("Translation Vector:\n", t)

    # Compute depth map using stereo triangulation
    disparity_map = compute_disparity_map(img1, img2)
    
    # Normalize for visualization
    disparity_map_normalized = cv2.normalize(disparity_map, None, alpha=0, beta=255,
                                             norm_type=cv2.NORM_MINMAX).astype(np.uint8)
    
    plt.subplot(1,3,3)
    plt.imshow(disparity_map_normalized, cmap='gray')
    plt.title("Disparity Map")
    plt.show()

    plt.imshow(disparity_map_normalized, cmap='gray')
    plt.title("Disparity Map")
    plt.colorbar()
    plt.show()
    return R, t, disparity_map

def compute_disparity_map(img1, img2):
    # Create StereoBM object and compute disparity map
    stereo = cv2.StereoBM_create(numDisparities=16*3, blockSize=11)
    disparity_map = stereo.compute(img1, img2)
    
    
    return disparity_map

# Example usage:
if __name__ == "__main__":
    # Replace with paths to your images and camera intrinsic matrix
    base_dir = "hills/"
    image1_path = base_dir+"image1.png"
    image2_path = base_dir+"image2.png"
    
    # Example camera intrinsic matrix (fx, fy assumed equal; cx and cy are principal points)
    camera_matrix = np.array([[700, 0, 640],
                               [0, 700, 360],
                               [0, 0, 1]])

    R, t, depth_map = compute_ego_motion_and_depth(image1_path, image2_path, camera_matrix)

