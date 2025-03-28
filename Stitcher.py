"""
Stitcher.py

This module provides functionality for stitching image fragments into whole images
using various blending techniques.

Author: Ing. David Pukanec
"""

import cv2 as cv
import numpy as np

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

    def warp_images(self, global_homographies, img_paths, pairs):
        """
        Warps images using global homographies  to align them to a common coordinate system.

        Args:
            global_homographies (list[np.ndarray]): List of global homographies for each image.
            img_paths (list[str]): List of file paths to the images.
            pairs (list[tuple[int, int]]): List of pairs of indices representing image relationships for stitching.

        Returns:
            dict[int, list[np.ndarray]]: A dictionary where the key is the image index, and the value is a list containing the warped image and its corresponding mask.
        """
        # Get the corner of the final image
        x_min, y_min = (0, 0)
        x_max, y_max = (self.config.data.final_res[1], self.config.data.final_res[0])

        # Compute translation homography to shift images to positive coordinates
        translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])
        # TODO [Refactor]
        # Iterate over the image paths and apply warping using the global homographies
        warped_list = {}
        for i, path in enumerate(img_paths):
            # Load the image
            img = cv.imread(path)
            # Calculate the homography
            H = translation @ global_homographies[i]
            # Apply warping based on homography
            warped = cv.warpPerspective(img, H, (x_max - x_min, y_max - y_min))
            # Mask
            mask = (warped > 0)
            warped_list[i] = [warped, mask]

        # Warp images based on pairs and apply the same translation and homography transformation
        for i, pair in enumerate(pairs):
            if 0 in pair:
                continue
            i += 1
            path = img_paths[pair[1]]
            img = cv.imread(path)
            H = translation @ global_homographies[i]
            # Apply warping based on homography
            warped = cv.warpPerspective(img, H, (x_max - x_min, y_max - y_min))
            mask = (warped > 0)
            warped_list[i] = [warped, mask]

        return warped_list


    def blend(self, blend_type, args):
        if blend_type == "basic":
            return self.blend_basic(args)
        elif blend_type == "weighted":
            return self.blend_weighted(args)
        elif blend_type == "actual":
            return self.blend_actual(args)

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


    def blend_actual(self, args):

        Hs = args['Hs']
        imgs = args['imgs']

        res = (int(self.config.data.final_res[0]), int(self.config.data.final_res[1]))
        idx_accum = np.ones(res) * -1
        val_accum = np.ones(res) * np.inf
        eroded_masks = {}
        for key, val in imgs.items():
            # Ignore reference
            if key == 0:
                continue
            img = val[0]
            mask = val[1]

            # Adjust the mask for overlap
            blend_width = self.config.stitcher.blend_width
            eros_kernel = np.ones((2 * blend_width +1, 2 * blend_width +1), np.uint8)
            shrunk_mask = cv.erode(mask.astype(np.uint8), eros_kernel, iterations=1).astype(bool)
            eroded_masks[key] = shrunk_mask

            y_min, x_min = np.argwhere(shrunk_mask[:,:,0]).min(axis=0) # Get min row and column
            y_max, x_max = np.argwhere(shrunk_mask[:,:,0]).max(axis=0)  # Get max row and column

            # Compute dense jacobian denterminants
            det = self.compute_jacobian_determinant(Hs[key], img[y_min:y_max, x_min:x_max].shape[:2])

            res_array = np.zeros(img.shape[:2])
            res_array[y_min:y_max, x_min:x_max] = det

            # Stack previous best values and current
            res_array[~shrunk_mask[:,:,0]] = np.inf
            stacked_val = np.stack([res_array, val_accum], axis=0)
            # compare them
            compare_idxs = np.abs(stacked_val - 1).argmin(axis=0)

            # select where the current was better
            accum_mask = compare_idxs == 0
            # Update accumulaotrs
            idx_accum[accum_mask & shrunk_mask[:,:,0]] = int(key)
            val_accum[accum_mask] = res_array[accum_mask]

        best_image_index = idx_accum
        weights = []

        for key, val in imgs.items():
            if key == 0:
                continue

            best_image_mask = best_image_index == int(key)

            weight = 1 - self.calc_border_dist(best_image_mask, k=50)[:, :, 0]
            weights.append(weight.astype(np.float16))


        list_images = [value[0] for key, value in imgs.items() if key not in [0]]

        final_image = self.blend_weighted_images(list_images, weights)
        cv.imwrite("./plots/weighted_img.jpg", final_image )
        return final_image

    def blend_weighted_images(self, images, weights):
        """
        Blends multiple images using per-pixel weights
        Args:
            images (list[np.ndarray]): List of images (H x W x C) to blend.
            weights (list[np.ndarray]): List of weights (H x W) to blend.

        Returns:
             np.ndarray: The final blended image (H x W x C) in uint8 format.
        """

        # Convert images to float16 for blending
        images = [img.astype(np.float32) for img in images]
        weights = [w.astype(np.float32) for w in weights]

        # Expand weight maps to match image channels (if grayscale)
        weights = [w[..., np.newaxis] if w.ndim == 2 else w for w in weights]

        # Compute the total weight per pixel (sum of all weights)
        total_weight = sum(weights)

        # Avoid division by zero (set total_weight to 1 where it's 0)
        total_weight = np.clip(total_weight, 1e-8, None)

        # Compute the final weighted sum
        weighted_sum = sum(img * w for img, w in zip(images, weights))

        # Normalize the final image
        final_image = weighted_sum / total_weight

        # Convert back to uint8
        final_image = np.clip(final_image, 0, 255).astype(np.uint8)

        return final_image

    def calc_border_dist(self, seam, k = 100, type=cv.THRESH_BINARY_INV):
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

    def blend_weighted(self, args):
        x_min, y_min = (0, 0)
        x_max, y_max = (self.config.data.final_res[1], self.config.data.final_res[0])
        stitched = np.zeros((y_max - y_min, x_max - x_min, 3), dtype=np.float32)
        acum = np.zeros((y_max - y_min, x_max - x_min, 3), dtype=np.uint8)

        for key, val in args.items():
            img = val[0]
            mask = val[1]
            # Do not include overlay img which is at index 0
            if key == 0:
                continue

            stitched[mask] = stitched[mask] + img[mask]
            acum[mask] += 1

        # Avoid division by 0
        acum_mask = acum > 0
        stitched[acum_mask] = stitched[acum_mask] / acum[acum_mask]
        if self.debug:
            cv.imwrite(f"./plots/stitched_weighted.jpg", stitched)
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






