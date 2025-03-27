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

        h, w = best_image_index.shape[:2]
        final_img = np.zeros((h, w, 3))
        final_mask = np.zeros((h, w), dtype=bool)
        cumulative_inter = np.zeros((h, w), dtype=int)

        weights = []
        list_images = []
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
        seam = np.array(seam * 255, dtype=np.uint8)
        _, thresh = cv.threshold(seam, 127, 255, type)
        seam_dist = cv.distanceTransform(thresh, cv.DIST_L2, 0)
        seam_dist = seam_dist / k
        seam_dist = np.minimum(seam_dist, 1)
        seam_dist = np.stack([seam_dist] * 3, axis=-1)
        return seam_dist

    def seam_gradient(self, best_image_index):
        mask = best_image_index == -1

        dx = np.zeros_like(best_image_index, dtype=np.float32)
        dx[:, :-1] = best_image_index[:, :-1] - best_image_index[:, 1:]

        dy = np.zeros_like(best_image_index, dtype=np.float32)
        dy[:-1, :] = best_image_index[:-1, :] - best_image_index[1:, :]

        # Set gradients to 0 wherever -1 was involved
        dx[mask] = 0
        dy[mask] = 0

        # Also set gradient to 0 where the next pixel is -1
        dx[:, :-1][best_image_index[:, 1:] == -1] = 0
        dy[:-1, :][best_image_index[1:, :] == -1] = 0

        # Compute masked derivative where both dx and dy are 0
        masked_derivative = np.where((dx == 0) & (dy == 0), 0, 1)

        # averaged_img = self.smooth_edges(best_image_index, masked_derivative, imgs)

        return masked_derivative


    def smooth_edges(self, image_mask, edge_mask, images, window_size=10):
        """
        image_mask: 2D array where each pixel contains an image index.
        edge_mask: Binary mask indicating where edges occur.
        images: List of RGB images corresponding to indices in the image mask.
        window_size: Size of the smoothing window.
        """
        pad = window_size // 2
        smoothed_image = np.zeros((6400, 8400, 3), dtype=np.float32)
        edge_coords = np.argwhere(edge_mask == 1)
        from tqdm import tqdm
        for y, x in tqdm(edge_coords):
            y_min, y_max = max(y - pad, 0), min(y + pad + 1, image_mask.shape[0])
            x_min, x_max = max(x - pad, 0), min(x + pad + 1, image_mask.shape[1])

            # Gather pixels from the corresponding images in the window
            img_indexes = np.unique(image_mask[y_min:y_max, x_min:x_max])
            for i in range(y_min, y_max):
                for j in range(x_min, x_max):
                    img_index = (image_mask[i, j])
                    # Extract the overlap mask of best image
                    mask = images[img_index][1]
                    averaging = []
                    # Iterate over all overlap masks to see contribution from other imgs
                    for key, val in images.items():
                        if mask[i,j,0] == val[1][i,j,0]  and key in img_indexes:
                            averaging.append(val[0][i,j])
                    smoothed_image[i,j] = np.mean(averaging, axis=0)

                    # window_pixels.append([i, j])

            # # Compute the mean of the window
            # if window_pixels:
            #     smoothed_image[y, x] = np.mean(window_pixels, axis=0)

        # Convert back to uint8
        smoothed_image = np.clip(smoothed_image, 0, 255).astype(np.uint8)
        return smoothed_image

    def blend_seam(self, image1, image2, seam_mask, blend_width=10):
        """
        Blends two images along a known seam using a smooth transition.

        Parameters:
            image1 (numpy.ndarray): First image (H, W, C).
            image2 (numpy.ndarray): Second image (H, W, C).
            seam_mask (numpy.ndarray): Binary mask (H, W) where 1 marks the seam edge.
            blend_width (int): Width of the transition region in pixels.

        Returns:
            numpy.ndarray: Blended image (H, W, C).
        """
        from scipy.ndimage import distance_transform_edt
        H, W, C = (6400,8400,3)

        img1 = image1[0]
        mask1 = image1[1]

        img2 = image2[0]
        mask2 = image2[1]

        # Compute distance transform from the seam (0 at seam, increasing outward)
        dist_to_seam = distance_transform_edt(1 - seam_mask)

        # Normalize to [0, 1] in the blend region (clipping at blend_width)
        weight_map = np.clip(dist_to_seam / blend_width, 0, 1)

        # Expand dimensions for broadcasting across color channels
        weight_map = weight_map[:, :, np.newaxis]

        # Adjust weights
        weight_map[mask1 and not mask2] = 0
        weight_map[mask2 and not mask1] = 1
        # weight_mask = n.pones_like(weight_map)
        # weight_map_cropped[mask1[:,:,0]] = 0
        # weight_map_cropped[mask1[:,:,0] and mask2[:,:,0]] = weight_map[mask1[:,:,0] and mask2[:,:,0]]
        # # Perform blending 1
        # blended_image1 = np.zeros((H, W, C)).astype(np.uint8)
        blended_image = (1 - weight_map) * img1 + weight_map[mask1[:,:,0]] * img2

        return blended_image

    def blend_images(self, images, index_map, blend_radius=5):
        """
        Blends images based on index map with soft transitions.

        Parameters:
            images (torch.Tensor): Tensor of shape (N, C, H, W) containing N images.
            index_map (torch.Tensor): Tensor of shape (H, W) with indices (0 to N-1).
            blend_radius (int): Radius for blending transition.

        Returns:
            torch.Tensor: Blended image of shape (C, H, W).
        """
        from scipy.ndimage import distance_transform_edt

        # Create mask for each image source
        masks = np.zeros((3, 6400, 8400))
        idx = 0
        for key, val in images.items():
            if key == 0:
                continue
            masks[idx] = (index_map == key)
            idx += 1

        # Compute distance transform for each mask
        distance_maps = np.zeros_like(masks)
        idx = 0
        for key, val in images.items():
            if key == 0:
                continue
            mask_np = masks[idx]
            distance_maps[idx] = distance_transform_edt(mask_np == 0)
            idx += 1

        # Compute blending weights (softmax-like normalization)
        weights = np.exp(-distance_maps / blend_radius)
        weights /= weights.sum(axis=0, keepdims=True)  # Normalize per pixel

        # Blend images using weights
        blended_image = (np.expand_dims(weights,1) * images).sum(axis=0)  # Weighted sum over images

        return blended_image

    def stitch_blend(self, arr):
        pass


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






