import os.path
import matplotlib.pyplot as plt
import cv2 as cv
import torch

from lightglue import LightGlue, SuperPoint, DISK
import numpy as np
class Stitcher():

    def __init__(self, config, debug=False):
        self.config = config
        self.debug = debug
        self.corners = None

    def get_overlap_region(self, img1, img2_warped):

        # Create binary masks for non-black regions
        mask1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY) > 0
        mask2 = cv.cvtColor(img2_warped, cv.COLOR_BGR2GRAY) > 0

        # Find the overlapping region
        overlap_mask = mask1 & mask2

        # Extract the overlapping areas
        overlap1 = cv.bitwise_and(img1, img1, mask=overlap_mask.astype(np.uint8))
        overlap2 = cv.bitwise_and(img2_warped, img2_warped, mask=overlap_mask.astype(np.uint8))

        return overlap1, overlap2

    def merges_images(self, imgA, imgB, H):

        height1, width1 = imgA.shape[:2]
        height2, width2 = imgB.shape[:2]

        canvas_width = max(width1, width2)
        canvas_height = max(height1, height2)

        warped_image = cv.warpPerspective(imgB, H, (canvas_width, canvas_height))

        # Stitch image into bigger frame
        empty_image = np.zeros_like(warped_image)
        # empty_image = np.zeros_like(imgA)
        empty_image[0:height1, 0:width1] = imgA
        stitched_image = np.maximum(empty_image, warped_image)
        # Blend the image
        alpha = 0.5
        overlap = (stitched_image > 0) & (warped_image > 0)
        # Overlap mask
        stitched_image[overlap] = (stitched_image[overlap] * alpha + warped_image[overlap] * (1 - alpha)).astype(
            np.uint8)

        if self.debug:
            cv.imwrite("./plots/warped.jpeg", warped_image)
            cv.imwrite("./plots/cv_stitched.jpg", stitched_image)

        overlap1, overlap2 = self.get_overlap_region(empty_image, warped_image)
        if self.debug:
            cv.imwrite("./plots/overlap1.jpg", overlap1)
            cv.imwrite("./plots/overlap2.jpg", overlap2)

        return stitched_image, [overlap1, overlap2]

    def get_corners(self, img_paths, global_homographies):

        corners = []

        for i, path in enumerate(img_paths):
            img = cv.imread(path)
            h, w = img.shape[:2]
            pts = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)  # Image corners
            pts_transformed = cv.perspectiveTransform(pts.reshape(-1, 1, 2), global_homographies[i])
            corners.append(pts_transformed.reshape(-1, 2))

        self.corners = corners
        return corners

    def warp_images(self, global_homographies, img_paths, pairs):

        # Get corners of the final image
        # corners = self.get_corners(img_paths, global_homographies)
        # corners = np.vstack(self.corners)
        # x_min, y_min = np.int32(corners.min(axis=0))
        # x_max, y_max = np.int32(corners.max(axis=0))
        x_min, y_min = (0, 0)
        x_max, y_max = (self.config.data.final_res[1], self.config.data.final_res[0])

        # Compute translation homography to shift images to positive coordinates
        translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])

        # Warp images
        #alpha = 0
        #stitched = np.zeros((y_max - y_min, x_max - x_min, 3), dtype=np.uint8)

        warped_list = {}
        for i, path in enumerate(img_paths):
            img = cv.imread(path)
            H = translation @ global_homographies[i]
            # Apply warping based on homography
            warped = cv.warpPerspective(img, H, (x_max - x_min, y_max - y_min))
            # Mask
            mask = (warped > 0)
            warped_list[i] = [warped, mask]



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

            # stitched[mask] = (stitched[mask] * alpha + warped[mask] * (1 - alpha))  # Simple blending
            if self.debug:
                cv.imwrite(f"plots/stitched{i}.jpg", warped)
        return warped_list


    def blend(self, blend_type, args):
        if blend_type == "basic":
            return self.blend_basic(args)
        elif blend_type == "weighted":
            return self.blend_weighted(args)
        elif blend_type == "actual":
            return self.blend_actual(args)

    def compute_jacobian_determinant(self, H, shape):
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

    def get_overlaping_idx(self, masks):
        pass

    def blend_actual(self, args):

        Hs = args['Hs']
        imgs = args['imgs']
        dets = []
        masks = []
        res = (int(self.config.data.final_res[0]), int(self.config.data.final_res[1]))
        idx_accum = np.zeros(res)
        val_accum = np.zeros(res) * np.inf
        for key, val in imgs.items():
            # Ignore reference
            if key == 0:
                continue
            #
            img = val[0]
            mask = val[1]
            y_min, x_min = np.argwhere(mask[:,:,0]).min(axis=0) # Get min row and column
            y_max, x_max = np.argwhere(mask[:,:,0]).max(axis=0)  # Get max row and column

            # Compute dense jacobian denterminants
            det = self.compute_jacobian_determinant(Hs[key], img[y_min:y_max, x_min:x_max].shape[:2])

            res_array = np.zeros(img.shape[:2])
            res_array[y_min:y_max, x_min:x_max] = det

            if self.config.debug:
                img_array = (res_array - res_array.min()) / (res_array.max() - res_array.min())  # Normalize to 0-1
                img_array = (img_array * 255).astype(np.uint8)  # Scale to 0-255 and convert to uint8
                cv.imwrite(f'plots/det_{key}.jpg', img_array)

            # dets.append(res_array)
            # masks.append(mask)
            res_array[~mask[:,:,0]] = np.inf
            stacked_val = np.stack([res_array, val_accum], axis=0)
            idx_accum = np.abs(stacked_val - 1).argmin(axis=0)
            val_accum = np.min(stacked_val, axis=0)
        # stacked_dets = np.stack(dets, axis=0)
        # stacked_masks = np.stack(masks, axis=0)
        #
        # stacked_dets[~stacked_masks[:,:,:,0]] = np.inf
        #
        # best_image_index = np.abs(stacked_dets - 1).argmin(axis=0) + 1
        best_image_index = idx_accum

        h, w = best_image_index.shape[:2]
        final_img = np.zeros((h, w, 3))
        for key, val in imgs.items():

            mask = best_image_index == int(key)
            final_img[mask] = val[0][mask]


        return final_img

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






