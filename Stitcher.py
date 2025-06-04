"""
Stitcher.py

This module provides functionality for stitching image fragments into whole images
using various blending techniques.

Author: Ing. David Pukanec
"""

import cv2 as cv
import numpy as np
from memory_profiler import profile

class Stitcher():
    """
        Main class holding the functionality for stitching fragments into whole images using different methods.
    """
    def __init__(self, config, debug=False):
        """
        Initialize the stitcher class.
        Args:
            config (DictConfig): Hydra configuration
            debug (bool): If debug mode is on save images to plots
        """
        self.config = config
        self.debug = debug

    def warp_image(self, homography, frag_path):
        """
        Warps image with homography.
        Args:
            homography:
            frag_path:

        Returns: warped fragment and mask

        """

        # Get the corner of the final image
        x_min, y_min = (0, 0)
        x_max, y_max = (self.config.data.final_res[1], self.config.data.final_res[0])
        # Compute translation homography to shift images to positive coordinates
        translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])

        # Load fragment
        fragment = cv.imread(frag_path)
        masking_array = np.ones_like(fragment, np.uint8)
        # Calculate the homography
        H = translation @ homography
        # Apply warping based on homography
        warped = cv.warpPerspective(fragment, H, (x_max - x_min, y_max - y_min))
        mask = cv.warpPerspective(masking_array, H, (x_max - x_min, y_max - y_min))
        # Mask
        mask = (mask > 0)

        return warped, mask


    def blend_weighted(self, args):

        if args['cache'] is not None:
            cache = args['cache']

        x_min, y_min = (0, 0)
        x_max, y_max = (self.config.data.final_res[1], self.config.data.final_res[0])
        stitched = np.zeros((y_max - y_min, x_max - x_min, 3), dtype=np.float32)
        acum = np.zeros((y_max - y_min, x_max - x_min, 3), dtype=np.uint8)

        for idx, val in enumerate(args["fragments"]):
            if args['cache'] is not None:
                img, mask = cache[val]
            else:
                img, mask = val

            stitched[mask] = stitched[mask] + img[mask]
            acum[mask] += 1

        # Avoid division by 0
        acum_mask = acum > 0
        stitched[acum_mask] = stitched[acum_mask] / acum[acum_mask]

        return stitched

    def blend_basic(self, args):

        # corners = np.vstack(self.corners)
        # x_min, y_min = np.int32(corners.min(axis=0))
        # x_max, y_max = np.int32(corners.max(axis=0))
        x_min, y_min = (0, 0)
        x_max, y_max = (self.config.data.final_res[1], self.config.data.final_res[0])

        alpha = 0.5
        stitched = np.zeros((y_max - y_min, x_max - x_min, 3), dtype=np.uint8)

        for key, val in args.items():
            img = val[0]
            mask = val[1]

            if key == 0:
                continue

            stitched[mask] = (stitched[mask] * alpha + img[mask] * (1 - alpha))
            alpha = 0.50
            if self.debug:
                cv.imwrite(f"./plots/stitched_{key}.jpg", stitched)

        return stitched

class DebugBlender:
    def __init__(self, size,config):
        self.config = config
        self.alpha = 0.5
        self.img = np.zeros((size[0], size[1], 3), dtype=np.float32)
        self.mask = np.zeros((size[0], size[1]), dtype=bool)

    def add_fragment(self, fragment, mask):

        fragment = fragment.astype(np.float32)
        weights = np.zeros_like(self.img)

        self.img[mask] += fragment[mask]
        weights[mask] += 1
        weights[self.mask] += 1

        acum_mask = weights > 0
        self.img[acum_mask] = self.img[acum_mask] / weights[acum_mask]

    def get_current_blend(self):
        return self.img.astype(np.uint8)


class ActualBlender:
    def __init__(self, config):
        res = (int(config.data.final_res[0]), int(config.data.final_res[1]))

        self.config = config
        self.blend_width = config.stitcher.blend_width
        # Erosion kerlnel for feathered edges
        self.erode_kernel = np.ones((2 * (self.blend_width + config.stitcher.flow_margin) + 1,
                                           2 * (self.blend_width + config.stitcher.flow_margin) + 1), np.uint8)

        # Progressive blend image values are accumulated here
        self.progress_blend_img = np.zeros((res[0], res[1], 3), dtype=np.float32)
        # Mask of progressive stitch
        self.progress_blend_mask = np.zeros(res, dtype=bool)
        # Accumulator for closest value to 1 this represent the best pixel so far
        self.progressive_val_accum = np.ones(res) * 99999
        self.best_idx_acum = np.ones(res) * -1


    def add_fragment(self, fragment, mask, homography, key):

        shrunk_mask = cv.erode(mask.astype(np.uint8), self.erode_kernel, iterations=1).astype(bool)
        if self.config.debug:
            cv.imwrite(f"./plots/shrunk_mask_{key}.jpg", shrunk_mask.astype(np.uint8) * 255)
        y_min, x_min = np.argwhere(shrunk_mask[:, :, 0]).min(axis=0)  # Get min row and column
        y_max, x_max = np.argwhere(shrunk_mask[:, :, 0]).max(axis=0)  # Get max row and column

        # Compute dense jacobian denterminants
        det = self.compute_jacobian_determinant(homography, fragment[y_min:y_max, x_min:x_max].shape[:2])
        res_array = np.zeros(fragment.shape[:2])
        res_array[y_min:y_max, x_min:x_max] = det

        # Stack previous best values and current
        res_array[~shrunk_mask[:, :, 0]] = 99999
        stacked_val = np.stack([res_array, self.progressive_val_accum], axis=0)
        # TODO Error accumulation
        compare_idxs = np.abs(np.log(np.abs(stacked_val) + 1e-8)).argmin(axis=0)
        # compare_idxs = np.abs(stacked_val - 1).argmin(axis=0)
        # Select where the current was better
        # frag_best_pixels_mask = (compare_idxs == 0) & shrunk_mask[:, :, 0]
        self.best_idx_acum[(compare_idxs == 0) & shrunk_mask[:, :, 0]] = key
        self.progressive_val_accum[compare_idxs == 0] = res_array[compare_idxs == 0]
        #shrunk_frag_best = cv.erode(frag_best_pixels_mask.astype(np.uint8), self.erode_kernel, iterations=1).astype(np.uint8)
        if self.config.debug:
            cv.imwrite(f"plots/frag_mask_{key}.jpg",((compare_idxs == 0) & shrunk_mask[:, :, 0]).astype(np.uint8) * 255)
            cv.imwrite(f"plots/best_idx_acum{key}.jpg",  self.best_idx_acum.astype(np.uint8) * 10)

        # Update current best pixel determinands

        cond = (self.best_idx_acum == key) | (self.best_idx_acum == -1)
        #prog_mask_eroded = cv.erode(self.progress_blend_mask.astype(np.uint8), self.erode_kernel, iterations=1).astype(bool)
        frag_weight = 1 - self.calc_border_dist(np.where(self.best_idx_acum == key, 1, 0), k=self.blend_width)[:, :, 0]
        prog_weight = 1 - self.calc_border_dist(np.where(cond, 0, 1), k=self.blend_width)[:, :, 0]
        #reverse_erosion = cv.dilate((frag_best_pixels_mask & shrunk_mask[:,:,0]).astype(np.uint8), self.erode_kernel, iterations=1)
        #frag_weight = self.calc_border_dist(reverse_erosion, k=self.blend_width, type=cv.THRESH_BINARY)[:, :, 0]

        if self.config.debug:
            cv.imwrite(f"plots/frag_weights_{key}.jpg", (frag_weight * 255).astype(np.uint8))
            cv.imwrite(f"plots/prog_weight_{key}.jpg", (prog_weight * 255).astype(np.uint8))


        # Given images and weights stitch them together
        self.progress_blend_img = self.blend_weighted([self.progress_blend_img, fragment],
                                                      [prog_weight, frag_weight])

        if self.config.debug:
            cv.imwrite(f"plots/prog_stitch_{key}.jpg", self.progress_blend_img.astype(np.uint8))
        # Update progressive mask
        self.progress_blend_mask = self.progress_blend_mask | mask[:,:,0]

    def get_current_blend(self):
        return self.progress_blend_img

    def compute_jacobian_determinant(self, H, shape):
        """
        Computes the Jacobian determinant for each pixel after applying homography H.
        Args:
            H (np.ndarray): 3x3 homography matrix.
            shape (tuple[int, int]): Shape (height, width) of the image.
        Return:
            np.ndarray: A 2D array (H x W) representing the Jacobian determinant at each pixel.
        """
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

    def calc_border_dist(self, seam, k=100, type=cv.THRESH_BINARY_INV):
        """
          Calculates the border distance transform for a mask.

          Args:
              seam (np.ndarray): Binary seam image (H x W), where the pixels are 1 or 0.
              k (int, optional): A scaling factor to normalize the distance in pixels. Default is 100.
              type (int, optional): The thresholding type used for creating the binary mask. Default is cv.THRESH_BINARY_INV.

          Returns:
              np.ndarray: A 3-channel image (H x W x 3) representing the border distance at each pixel.
          """
        seam = np.array(seam * 255, dtype=np.uint8)
        #  Apply thresholding to create a binary mask (invert thresholding by default)
        _, thresh = cv.threshold(seam, 127, 255, type)
        # Compute the distance transform to get the distance from the nearest seam pixel
        seam_dist = cv.distanceTransform(thresh, cv.DIST_L2, 0)
        # Normalize the distance map
        seam_dist = seam_dist / k
        seam_dist = np.minimum(seam_dist, 1)
        # Expand it into img dims
        seam_dist = np.stack([seam_dist] * 3, axis=-1)

        return seam_dist

    def blend_weighted(self, imgs, weights):

        total_weight = np.sum(weights, axis=0)
        res_img = np.zeros_like(imgs[0])
        for idx, img in enumerate(imgs):

            weight = weights[idx][:, :, np.newaxis]
            # Apply weight  to the image
            weighted_img = img * weight
            res_img += weighted_img

        # Avoid division by zero
        nonzero_mask = total_weight > 0

        # Expand dims to match shape (H, W, 3)
        total_weight_expanded = np.where(
            nonzero_mask[:, :, np.newaxis],
            total_weight[:, :, np.newaxis],
            1  # prevent division by zero
        )

        res_img /= total_weight_expanded
        return res_img

