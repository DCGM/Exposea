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
    The Stitcher class is designed to handle image stitching operations.

    This class provides methods for warping and blending images. By utilizing
    the configurations provided during initialization, the Stitcher class can
    process images efficiently and optionally save debug outputs. Users can
    use the implemented warping and blending techniques to generate final
    stitched images based on input parameters and configurations.
    """
    def __init__(self, config, debug=False):

        self.config = config
        self.debug = debug

    def warp_image(self, homography, frag_path, res=None):
        """
        Warp the given image fragment using a homography matrix.

        This function applies a homography transformation to warp an image fragment
        into a specific coordinate space. The resulting warped image and a corresponding
        mask are returned. The mask indicates valid regions within the warped image.
        If no resolution is provided, the function uses the default final resolution
        from configuration, otherwise it uses the specified resolution.

        Parameters:
            homography: numpy.ndarray
                A 3x3 homography matrix for warping the image.
            frag_path: str
                The path to the image fragment to be warped.
            res: tuple[int, int], optional
                The target resolution (height, width) for the warped image. Defaults to None.

        Returns:
            tuple[numpy.ndarray, numpy.ndarray]
                The warped image as a numpy array and its corresponding mask.
        """
        # Get the corner of the final image

        x_min, y_min = (0, 0)
        if res is None:
            x_max, y_max = (self.config.final_res[1], self.config.final_res[0])
        else:
            x_max, y_max = (res[1], res[0])
        # Compute translation homography to shift images to positive coordinates
        translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])

        # Load fragment
        fragment = cv.imread(frag_path).astype(np.float32)
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
        x_max, y_max = (self.config.final_res[1], self.config.final_res[0])
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
        x_max, y_max = (self.config.final_res[1], self.config.final_res[0])

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
    """
    A class for blending image fragments with a mask to form a cumulative blend.

    This class allows for the addition of image fragments and their respective masks
    to a blending process, accumulating results while taking weights into account.

    Attributes:
        config: Configuration for the blending process.
        alpha: Blend factor used in processing.
        img: Numpy array holding the current blended image.
        mask: Boolean numpy array indicating the accumulated blending mask.

    Methods:
        add_fragment(fragment, mask):
            Adds a new image fragment to the blend using the provided mask.
        get_current_blend():
            Retrieves the current blended image as an unsigned 8-bit integer numpy array.
    """
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
        self.mask = acum_mask[:, :, 0]

    def get_current_blend(self):
        return self.img.astype(np.uint8)


class ActualBlender:
    """
    Represents a class for progressive blending of image fragments using
    homographies and weighted blending techniques.

    The ActualBlender class is designed for stitching and blending image
    fragments with progressive updates, leveraging homography-based
    transformations. It manages the blending process with configurable settings
    including an erosion kernel, blend width, and other progressive blending
    mechanisms.

    Attributes:
        config (ConfigType): Configuration object for stitching and blending.
        blend_width (int): Width for applying feathering blend techniques.
        erode_kernel (np.ndarray): Kernel used for morphological erosion during mask processing.
        progress_blend_img (np.ndarray): Accumulator image for progressively blended fragments.
        progress_blend_mask (np.ndarray): Boolean mask identifying which areas have been blended.
        progressive_val_accum (np.ndarray): Accumulator for the minimum determinant values, representing
            the best blending pixels.
        best_idx_acum (np.ndarray): Array storing the identifier of the fragment contributing
            the best pixel for each position.

    Methods:
        add_fragment(fragment: np.ndarray, mask: np.ndarray, homography: np.ndarray, key: int):
            Adds a new image fragment to the progressive blending result.

        get_current_blend() -> np.ndarray:
            Retrieves the current blended image.

        compute_jacobian_determinant(H: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
            Computes the Jacobian determinant for each pixel after applying a homography.

        calc_border_dist(seam: np.ndarray, k: int = 100, type: int = cv.THRESH_BINARY_INV) -> np.ndarray:
            Calculates the border distance transform for a binary mask.

        blend_weighted(imgs: list[np.ndarray], weights: list[np.ndarray]) -> np.ndarray:
            Blends multiple weighted images together and returns the result.
    """
    def __init__(self, config, ref_img):
        res = (int(config.final_res[0]), int(config.final_res[1]))

        self.config = config
        self.blend_width = config.stitcher.blend_width
        # Erosion kerlnel for feathered edges
        self.erode_kernel = np.ones((2 * (self.blend_width + config.stitcher.flow_margin) + 1,
                                           2 * (self.blend_width + config.stitcher.flow_margin) + 1), np.uint8)

        # Progressive blend image values are accumulated here
        res_ref = cv.resize(ref_img, (res[1], res[0]), interpolation=cv.INTER_AREA)
        self.progress_blend_img = np.array(res_ref, dtype=np.float32)
        del res_ref
        # Mask of progressive stitch
        self.progress_blend_mask = np.zeros(res, dtype=bool)
        # Accumulator for closest value to 1 this represent the best pixel so far
        self.progressive_val_accum = np.ones(res) * 99999
        self.best_idx_acum = np.ones(res) * -1


    def add_fragment(self, fragment, mask, homography, key):
        """
        Add a new fragment to the image blend while updating masks and accumulators.
        Applies erosion to the mask, calculates jacobian determinants, determines the
        best values for blending, and updates relevant accumulation arrays. Optionally
        saves debug images at each step if debugging is enabled within the configuration.

        Parameters:
            fragment: np.ndarray
                The fragment image to be added to the blend.
            mask: np.ndarray
                The binary mask indicating the valid region of the fragment.
            homography: np.ndarray
                The homography matrix for transforming the fragment.
            key: int
                The unique identifier for the current fragment.

        """
        shrunk_mask = cv.erode(mask.astype(np.uint8), self.erode_kernel, iterations=1).astype(bool)
        if self.config.debug:
            cv.imwrite(f"./plots/shrunk_mask_{key}.jpg", shrunk_mask.astype(np.uint8) * 255)
        y_min, x_min = np.argwhere(shrunk_mask[:, :, 0]).min(axis=0)  # Get min row and column
        y_max, x_max = np.argwhere(shrunk_mask[:, :, 0]).max(axis=0)  # Get max row and column

        # Compute dense jacobian denterminants
        det = self.compute_jacobian_determinant(homography, fragment[y_min:y_max, x_min:x_max].shape[:2])
        res_array = np.zeros(fragment.shape[:2], dtype=np.float32)
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

        if self.config.debug:
            cv.imwrite(f"plots/frag_weights_{key}.jpg", (frag_weight * 255).astype(np.uint8))
            cv.imwrite(f"plots/prog_weight_{key}.jpg", ((1 - frag_weight) * 255).astype(np.uint8))


        # Given images and weights stitch them together
        self.progress_blend_img = self.blend_weighted([self.progress_blend_img, fragment],
                                                      [1 - frag_weight, frag_weight])

        if self.config.debug:
            cv.imwrite(f"plots/prog_stitch_{key}.jpg", self.progress_blend_img.astype(np.uint8))
        # Update progressive mask
        self.progress_blend_mask = self.progress_blend_mask | mask[:,:,0]

    def get_current_blend(self):
        return self.progress_blend_img

    def compute_jacobian_determinant(self, H, shape):
        """
        Computes the determinant of the Jacobian matrix for a given homography transformation and image shape.

        The function calculates the determinant of the Jacobian matrix for each pixel,
        determining local distortions introduced by the homography when mapping pixel
        coordinates from one image to another.

        Parameters:
        H : array-like of shape (3, 3)
            The homography transformation matrix.
        shape : tuple
            A tuple representing the dimensions of the image (height, width), where the
            Jacobian determinant is calculated.

        Returns:
        array
            A 2D array containing the determinant of the Jacobian matrix for each pixel
            coordinate in the output image.
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
        Calculates a normalized distance map from the seam to the nearest seam pixel
        using distance transform. The resulting map represents normalized distances
        that can be used as a measure of proximity.

        Parameters:
            seam (np.ndarray): A binary array indicating the seam where proximity is
                calculated. Each element should be in the range of 0 to 1.
            k (int, optional): Normalization factor for the distance map. Higher values
                make the distances more granular. Default is 100.
            type (int, optional): Type of thresholding applied to the seam for creating
                the binary mask. Default is cv.THRESH_BINARY_INV.

        Returns:
            np.ndarray: A distance map normalized to values between 0 and 1. The map
                is in the same dimensions as the image with an additional axis
                replicated 3 times to match RGB channels.
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
        """
        Blends multiple images using per-pixel weights. Each image is multiplied by its
        corresponding weight, and the weighted images are summed together. The result
        is normalized by the sum of the weights, avoiding division by zero.

        Args:
            imgs (List[np.ndarray]): A list of numpy arrays representing the images to
                be blended. Each image must have the same shape.
            weights (np.ndarray): A 3D numpy array of weights used for blending, with
                shape (N, H, W), where N is the number of images, and H and W match the
                dimensions of each image in the list.

        Returns:
            np.ndarray: A numpy array representing the blended image with the same
                shape as each image in `imgs`.
        """
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

