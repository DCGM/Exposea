<<<<<<< HEAD

import pickle
import numpy as np
import cv2 as cv
if __name__ == "__main__":

    def compute_jacobian_determinant(H, shape):
        """Computes the Jacobian determinant for each pixel after applying homography H."""
        h, w = shape[:2]

        # Generate a grid of pixel coordinates
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        ones = np.ones_like(x)

        # Convert to homogeneous coordinates
        coords = np.stack([x, y, ones], axis=-1).reshape(-1, 3).T  # Shape: (3, N)

        # Apply homography to get transformed coordinates
        transformed_coords = H @ coords
        transformed_coords /= transformed_coords[2]  # Normalize homogeneous coordinates

        x_prime = transformed_coords[0].reshape(h, w)
        y_prime = transformed_coords[1].reshape(h, w)

        # Compute partial derivatives to get the Jacobian matrix
        dx_dx, dx_dy = np.gradient(x_prime, axis=(0, 1))  # Partial derivatives of x'
        dy_dx, dy_dy = np.gradient(y_prime, axis=(0, 1))  # Partial derivatives of y'

        # Compute determinant of the Jacobian matrix at each pixel
        J_det = dx_dx * dy_dy - dx_dy * dy_dx  # det(J)

        return J_det

    def comp_jacobian(H, shape):
        h, w = shape[:2]
        # Create grid of (x, y) coordinates
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        ones = np.ones_like(x)
        coords = np.stack([x, y, ones], axis=-1).reshape(-1, 3).T

        # Apply homography to get (x', y', w')
        warped = H @ coords
        w_prime = warped[2]
        x_prime = warped[0] / w_prime
        y_prime = warped[1] / w_prime

        # Compute partial derivatives numerically (finite differences)
        dx = 1e-5  # Small step for numerical derivative
        coords_dx = np.stack([x + dx, y, ones], axis=-1).reshape(-1, 3).T
        warped_dx = H @ coords_dx
        x_prime_dx = warped_dx[0] / warped_dx[2]
        y_prime_dx = warped_dx[1] / warped_dx[2]

        coords_dy = np.stack([x, y + dx, ones], axis=-1).reshape(-1, 3).T
        warped_dy = H @ coords_dy
        x_prime_dy = warped_dy[0] / warped_dy[2]
        y_prime_dy = warped_dy[1] / warped_dy[2]

        # Jacobian components
        dx_prime_dx = (x_prime_dx - x_prime) / dx
        dx_prime_dy = (x_prime_dy - x_prime) / dx
        dy_prime_dx = (y_prime_dx - y_prime) / dx
        dy_prime_dy = (y_prime_dy - y_prime) / dx

        # Jacobian determinant
        det_J = dx_prime_dx * dy_prime_dy - dx_prime_dy * dy_prime_dx
        return det_J

    with open("opt_hom_20250228_164634.pkl", "rb") as f:
        homographies = pickle.load(f)
    H =  homographies[13]
    shape = (600,800)

    J_det2 = comp_jacobian(H, shape)
    J_det = compute_jacobian_determinant(H, shape)
    img_array = (J_det - J_det.min()) / (J_det.max() - J_det.min())  # Normalize to 0-1
    img_array = (img_array * 255).astype(np.uint8)  # Scale to 0-255 and convert to uint8

    # Save to file
    cv.imwrite('output_image_cv2.png', img_array)
=======
from lightglue import LightGlue, SuperPoint, DISK
from lightglue.utils import load_image, rbd
from lightglue import viz2d
import torch
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from stitching.images import Images
from stitching.subsetter import Subsetter
from Stitcher import Stitcher
from Optical import OpticalFlow
from HomogEst import HomogEstimator

#
# IMAGE1_P = "imgs/_MG_9344.JPG"
# IMAGE2_P = "imgs/_MG_9343.JPG"
#
#
# def save_debug_imgs(dat1, dat2, matches, path="./plots/"):
#
#     axes = viz2d.plot_images([dat1[0], dat2[0]])
#     viz2d.plot_matches(dat1[1], dat2[1], color="lime", lw=0.2)
#     viz2d.add_text(0, f'Stop after {matches["stop"]} layers', fs=20)
#     viz2d.save_plot(path+ "matches.jpg")
#     kpc0, kpc1 = viz2d.cm_prune(matches["prune0"]), viz2d.cm_prune(matches["prune1"])
#     viz2d.plot_images([dat1[0], dat2[0]])
#     viz2d.plot_keypoints([dat1[2], dat2[2]], colors=[kpc0, kpc1], ps=10)
#     viz2d.save_plot(path + "keypoints.jpg")
#
# def merges_images(imgA, imgB, H):
#
#     height1, width1 = imgA.shape[:2]
#     height2, width2 = imgB.shape[:2]
#
#     canvas_width = width1 + width2
#     canvas_height = max(height1, height2)
#
#     warped_image = cv.warpPerspective(imgB, H, (canvas_width, canvas_height))
#
#     # Stitch image into bigger frame
#     empty_image = np.zeros_like(warped_image)
#     empty_image[0:height1, 0:width1] = imgA
#     stitched_image = np.maximum(empty_image, warped_image)
#     # Blend the image
#     alpha = 0.5
#     overlap = (stitched_image > 0) & (warped_image > 0)
#     # Overlap mask
#     stitched_image[overlap] = (stitched_image[overlap] * alpha + warped_image[overlap] * (1 - alpha)).astype(np.uint8)
#
#     cv.imwrite("./plots/warped.jpeg", warped_image)
#     cv.imwrite("./plots/cv_stitched.jpg", stitched_image)
#
#     overlap1, overlap2 = stitcher.get_overlap_region(empty_image, warped_image)
#
#     cv.imwrite("./plots/overlap1.jpg", overlap1)
#     cv.imwrite("./plots/overlap2.jpg", overlap2)
#
#     return stitched_image, [overlap1, overlap2]
#
#
# if __name__ == "__main__":
#
#     # Get features extractor
#     stitcher = Stitcher()
#
#     lg_img1 = load_image(IMAGE1_P)
#     lg_img2 = load_image(IMAGE2_P)
#
#     # Extract features for matching
#     feats1 = stitcher.get_features(lg_img1)
#     feats2 = stitcher.get_features(lg_img2)
#
#     # Match features for homography estimation
#     matches12 = stitcher.match_feat_pairs(feats1, feats2)
#
#     feats1, feats2, matches12 = [
#         rbd(x) for x in [feats1, feats2, matches12]
#     ]  # remove batch dimension
#     # Extract Matched key points from key points lists for homography
#     kpts1, kpts2, matches = feats1["keypoints"], feats2["keypoints"], matches12["matches"]
#     m_kpts1, m_kpts2 = kpts1[matches[..., 0]], kpts2[matches[..., 1]]
#
#     save_debug_imgs([lg_img1, m_kpts1, kpts1], [lg_img2, m_kpts2, kpts2], matches12)
#
#     # Get homography
#     H, mask = cv.findHomography(np.asarray(m_kpts2.cpu()), np.asarray(m_kpts1.cpu()), cv.RANSAC, 5.0)
#
#     cv_img1 = cv.imread(IMAGE1_P)
#     cv_img2 = cv.imread(IMAGE2_P)
#     # Merged images with overlap masks from the original one
#     merged_image, overlap_masks = merges_images(cv_img1, cv_img2, H)
#
#     # Crop overlap to feed it into Optical flow
#     _, bin_mask1 = cv.threshold(cv.cvtColor(overlap_masks[0], cv.COLOR_BGR2GRAY), 1, 255, cv.THRESH_BINARY)
#     _, bin_mask2 = cv.threshold(cv.cvtColor(overlap_masks[1], cv.COLOR_BGR2GRAY), 1, 255, cv.THRESH_BINARY)
#
#     coords1 = cv.findNonZero(bin_mask1)
#     coords2 = cv.findNonZero(bin_mask2)
#
#     x, y, w, h = cv.boundingRect(coords1)
#     cropped_image1 = overlap_masks[0][y:y+h, x:x+w]
#
#     x, y, w, h = cv.boundingRect(coords2)
#     cropped_image2 = overlap_masks[1][y:y + h, x:x + w]
#
#     cv.imwrite("./plots/cropped1.jpg", cropped_image1)
#     cv.imwrite("./plots/cropped2.jpg", cropped_image2)
#
#     # Resize the img to computable size for optical flow
#     size = (960, 432)
#     resized_img = [cv.resize(cropped_image1, size), cv.resize(cropped_image2, size)]
#     cv.imwrite("./plots/resized1.jpg", resized_img[0])
#     cv.imwrite("./plots/resized2.jpg", resized_img[1])
#     # Optical flow
#     raft = OpticalFlow('sea_raft_m', 'kitti')
#     raft.upload_images(resized_img)
#     opt_flow = raft.flow()
#     flows = opt_flow['flows']
#
#     flow_bgr = raft.flow_to_img(flows)
#     cv.imwrite("./plots/flow.jpg", (flow_bgr * 255).astype('uint8'))
#     _, bin_mask_flow = cv.threshold(cv.cvtColor(resized_img[0], cv.COLOR_BGR2GRAY), 1, 255, cv.THRESH_BINARY)
#
#     masked_flow = cv.bitwise_and(flow_bgr, flow_bgr, mask=bin_mask_flow)
#     cv.imwrite("./plots/masked_flow.jpg", (masked_flow * 255).astype('uint8'))
#
#     #flow_np = flows[0, 0].permute(2, 1, 0).cpu().numpy()
#     resized_flow = cv.resize(masked_flow, cropped_image2.shape[:2])
#     cv.imwrite("./plots/resized_flow.jpg", (resized_flow * 255).astype('uint8'))
#
#     h, w = cropped_image1.shape[:2]
#     grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
#     flow_map_x = (grid_x + resized_flow[..., 0]).astype(np.float32)
#     flow_map_y = (grid_y + resized_flow[..., 1]).astype(np.float32)
#
#     aligned_overlap2 = cv.remap(cropped_image2, flow_map_x, flow_map_y, cv.INTER_LINEAR)
#     cv.imwrite("./plots/distorted.jpg", (aligned_overlap2 * 255).astype('uint8'))

if __name__ == "__main__":
    ov = cv.imread("./imgs/example1/0.JPG")
    ov = cv.resize(ov, (8400, 6400))
    cv.imwrite("./imgs/example1/0.JPG", ov)
