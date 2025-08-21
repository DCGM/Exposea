import copy
import logging

import torch
import ptlflow
from ptlflow.utils import flow_utils
from ptlflow.utils.io_adapter import IOAdapter
import cv2 as cv
import os
import numpy as np
from tqdm import tqdm
from memory_profiler import profile
import torch.profiler
import sys



class OpticalFlow:

    def __init__(self, config):
        self.logger = logging.getLogger("OPTICAL FLOW")
        self.config = config
        self.model = ptlflow.get_model(config.optical.model, config.optical.checkpoint)
        self.model.training = False
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.io_adapter = IOAdapter(self.model, config.optical.input_size, cuda=True)

        self.debug_idx = 0
    def get_input(self, images):
        """
        Helper function to get input format for flow estimation
        Args:
            images: list of two images
        Returns:

        """

        # This is basic function for converting to input of batch size 1
        input_i = self.io_adapter.prepare_inputs(images)
        return input_i

    def flow(self, input):
        """
        Estimates optical flow using flow estimation model
        Args:
            input: input images

        Returns: flow predictions

        """
        # Forward the inputs through the model
        predictions = self.model(input)
        return predictions

    def torch_flow_to_img(self, flows):
        """
        Converts a flow tensor into an image in BGR format suitable for visualization.
        Args
            flows: A tensor representing optical flow. Shape: (batch_size, channels, height, width).

        Returns: A NumPy array representing the flow in BGR format, ready for display using OpenCV.

        """
        flow_img = flow_utils.flow_to_rgb(flows)
        flow_rgb = flow_img[0, 0].permute(1, 2, 0)
        flow_rgb_npy = flow_rgb.detach().cpu().numpy()
        # OpenCV uses BGR format
        flow_bgr_npy = cv.cvtColor(flow_rgb_npy, cv.COLOR_RGB2BGR)

        return flow_bgr_npy

    def np_flow_to_img(self, flows):
        """
             Converts a flow np array into an image in BGR format suitable for visualization.
             Args
                 flows: A np array representing optical flow. Shape: (height, width, channels).

             Returns: A NumPy array representing the flow in BGR format, ready for display using OpenCV.

             """
        flow_img = flow_utils.flow_to_rgb(flows)
        # OpenCV uses BGR format
        flow_bgr_npy = cv.cvtColor(flow_img, cv.COLOR_RGB2BGR)

        return flow_bgr_npy

    def estimate_patches(self, patches):
        """
        Runs optical flow on patches of given size
        Args:
            patches: Individual patches of size HxW given by input_size

        Returns:
            list of estimated patches
        """
        # For saving resulting flows
        flow_patches = []
        idx = 0
        # Progress bar
        patch_pbar = tqdm(total=len(patches), desc='Processing patches',
                          position=1, leave=False, ncols=100, colour='red', file=sys.stdout)
        # TODO Rework to batches
        # Iterate over all patches
        self.logger.info(f'Estimating flow on {len(patches)} patches')
        for patch in patches:
            idx += 1
            # We reshape so that the estimation is from fragment to overview
            reshaped_patch = (patch[0], patch[1])
            # Check if both images have same shape
            if patch[0].shape[:2] != patch[1].shape[:2]:
                self.logger.error(f"Patch shape missmatch {patch[0].shape[:2]} != {patch[1].shape[:2]}")
            assert patch[0].shape[:2] == patch[1].shape[:2]
            # Find mask of patch optical flow on black background
            _, bin_mask_p = cv.threshold(cv.cvtColor(patch[0], cv.COLOR_BGR2GRAY), 1, 255, cv.THRESH_BINARY)

            #Estimate flow
            input_i = self.get_input(reshaped_patch)
            opt_flow = self.flow(input_i)
            # Extract only flow information
            flows = opt_flow['flows']

            # Get the flows into usable format
            flow = flows[0, 0].permute(1, 2, 0).detach().cpu().numpy()
            # Remove flow from non image pixels
            mask = np.expand_dims(bin_mask_p > 0.5, axis=-1)
            flow = flow * mask.astype(flow.dtype)

            # Append later for return
            flow_patches.append(flow)

            patch_pbar.update(1)

        return flow_patches

    def estimate_flow(self, ref_img, frag_img, debug_idx=0):

        # Get the overlapping region  for optical flow estimation
        overlap1, overlap2 = self.get_overlap_region(ref_img, frag_img)
        assert overlap1.shape == overlap2.shape

        _, bin_mask_ov = cv.threshold(cv.cvtColor(overlap1, cv.COLOR_BGR2GRAY), 1, 255, cv.THRESH_BINARY)

        coords_ov = cv.findNonZero(bin_mask_ov)

        o_x, o_y, o_w, o_h = cv.boundingRect(coords_ov)
        cropped_ov = overlap1[o_y:o_y + o_h, o_x:o_x + o_w]
        cropped_fr = overlap2[o_y:o_y + o_h, o_x:o_x + o_w]

        if self.config.optical.debug:
            cv.imwrite(f"./plots/cropped_ov_{debug_idx}.jpg", cropped_ov)
            cv.imwrite(f"./plots/cropped_fr_{debug_idx}.jpg", cropped_fr)

        if cropped_fr.shape != cropped_ov.shape:
            raise ValueError("Images must have the same shape.")

        # Subsample and blur the fragment for matching the resolutions
        if self.config.optical.adjust:
            resized_imgs = self.adjust_images(cropped_ov, cropped_fr, self.config)
        else:
            resized_imgs = [cropped_ov, cropped_fr]

        # Get the size of patch | [height, width]
        patch_size = self.config.optical.input_size


        # Returns patches list of tuples [(overview, fragment)] patches and relative position of the patch
        patches, positions = self.split_image_with_overlap(resized_imgs, patch_size,
                                                           self.config.optical.patch_overlap)
        # Estimate the flow on patches
        flow_patches = self.estimate_patches(patches)

        stitched_flow = self.merge_flows(flow_patches, positions, resized_imgs[0].shape,
                                         self.config.optical.patch_overlap)
        # # If subsampled then upsample to original
        # if resized_imgs[0].shape != cropped_fr.shape:
        #     stitched_flow = cv.resize(stitched_flow, (cropped_fr.shape[1], cropped_fr.shape[0]), interpolation=cv.INTER_LINEAR)
        if hasattr(self.config.optical, 'adjust_params') and hasattr(self.config.optical.adjust_params, 'scale_factor'):
            scale = self.config.optical.adjust_params.scale_factor
            stitched_flow = cv.resize(stitched_flow, (stitched_flow.shape[1] * scale, stitched_flow.shape[0] * scale),
                                      interpolation=cv.INTER_AREA)
            o_y *= scale
            o_x *= scale
            o_h *= scale
            o_w *= scale
            ref_img = cv.resize(ref_img, (ref_img.shape[1] * scale, ref_img.shape[0] * scale),
                                      interpolation=cv.INTER_LINEAR)

        flow_in_ov = np.zeros((ref_img.shape[0], ref_img.shape[1], 2))
        # Get the flow into dimension and position of overlap image
        flow_in_ov[o_y:o_y + o_h, o_x:o_x + o_w] = stitched_flow[:o_h, :o_w]

        if self.config.debug:
            stitched_flow_img = self.np_flow_to_img(stitched_flow)
            cv.imwrite(f"./plots/stitched_flow_{debug_idx}.jpg", stitched_flow_img)
            flow_in_ov_img = self.np_flow_to_img(flow_in_ov)
            cv.imwrite(f"./plots/flow_in_ov_img_{debug_idx}.jpg", flow_in_ov_img)

        #logging.warning(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))
        return flow_in_ov


    def adjust_images(self, reference, fragment, config):

        if hasattr(config, 'relative_scale'):
            height, width = fragment.shape[0],  fragment.shape[1]
            scale = 1 / config.relative_scale
            new_size = (int(width * scale), int(height * scale))
            ds_frag = cv.resize(fragment, new_size, interpolation=cv.INTER_AREA)
            ds_frag = cv.GaussianBlur(ds_frag, (3, 3), sigmaX=0.5)
            ds_frag = cv.resize(ds_frag, (width, height), interpolation=cv.INTER_CUBIC)
            ds_ref = reference
        else:
            ds_frag = fragment
            ds_ref = reference

        if config.debug:
            cv.imwrite(f"./plots/subsampled_ref{self.debug_idx}.jpg", ds_ref)
            cv.imwrite(f"./plots/subsampled_frag{self.debug_idx}.jpg", ds_frag)
        self.debug_idx += 1

        return ds_ref, ds_frag

    def get_coords_from_mask(self, maskA, maskB):

        _, bin_mask_ov = cv.threshold(cv.cvtColor(maskA.astype(np.uint8), cv.COLOR_BGR2GRAY), 1, 255, cv.THRESH_BINARY)
        _, bin_mask_fr = cv.threshold(cv.cvtColor(maskB.astype(np.uint8), cv.COLOR_BGR2GRAY), 1, 255, cv.THRESH_BINARY)

        coords_ov = cv.findNonZero(bin_mask_ov)
        coords_fr = cv.findNonZero(bin_mask_fr)

        return coords_fr, coords_ov

    def split_image_with_overlap(self, images, patch_size, overlap):
        image_a, image_b = images
        h, w, c = image_a.shape
        ph, pw = patch_size
        oh, ow = overlap

        core_ph = ph - 2 * oh
        core_pw = pw - 2 * ow

        patches = []
        positions = []

        for y in range(0, h, core_ph):
            for x in range(0, w, core_pw):
                # Take care of patch exiting image boundaries
                if y + core_ph >= h:
                    y = h - core_ph
                if x + core_pw >= w:
                    x = w - core_pw

                # Take care of patch with overlap exiting image boundaries
                y0 = max(y - oh, 0)
                x0 = max(x - ow, 0)
                y1 = min(y + core_ph + oh, h)
                x1 = min(x + core_pw + ow, w)

                patch_a = image_a[y0:y1, x0:x1]
                patch_b = image_b[y0:y1, x0:x1]

                patches.append((patch_a, patch_b))
                positions.append((y, x))

        return patches, positions

    def merge_flows(self, flows, positions, original_shape, overlap):
        img_h, img_w, _ = original_shape
        oh, ow = overlap
        p_h, p_w = self.config.optical.input_size

        core_ph = p_h - 2 * oh
        core_pw = p_w - 2 * ow

        merged_flow = np.zeros((img_h, img_w, 2), dtype=np.float32)
        merged_acc = np.zeros((img_h, img_w, 2), dtype=np.float32) + 1e-5

        for idx, (patch, (y, x)) in enumerate(zip(flows, positions)):
            # Patch overlap outside of image do not include it
            if y == 0:
                patch = patch[:core_ph + oh, :]
            elif y + core_ph + oh >= img_h:
                patch = patch[oh:core_ph, :]
            else:
                patch = patch[oh:core_ph + oh, :]

            if x == 0:
                patch = patch[:, :core_pw + ow]
            elif x + core_pw + ow >= img_w:
                patch = patch[:, ow:core_pw]
            else:
                patch = patch[:, ow:core_pw + ow]

            ph, pw = patch.shape[:2]

            merged_flow[y:y + ph, x:x + pw] += patch
            merged_acc[y:y + ph, x:x + pw] += 1

        normalized_flow = merged_flow / merged_acc
        return normalized_flow


    def warp_mask(self, image, flow):
        """
        Applies optical flow to image
        Args:
            image:
            flow:

        Returns:

        """
        h, w = flow.shape[:2]

        # Create mesh grid of pixel indices
        y, x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')

        # Compute new pixel positions
        x_new = np.clip(x + flow[..., 0], 0, w - 1)
        y_new = np.clip(y + flow[..., 1], 0, h - 1)

        # Warp image using remap
        warped = cv.remap(copy.copy(image), x_new.astype(np.float32), y_new.astype(np.float32),
                          interpolation=cv.INTER_NEAREST, borderMode=cv.BORDER_REPLICATE)
        return warped



    def warp_image_tiled(self,image, flow, tile=4096,
                   interpolation="linear",
                   border_mode="replicate",
                   border_value=0,
                   dtype_out=None):
        """
        Memory-efficient tiled backward warping of an image using optical flow.

        Parameters
        ----------
        image : np.ndarray
            Input image of shape (H, W) or (H, W, C).
        flow : np.ndarray
            Backward optical flow of shape (H, W, 2). flow[...,0] = u (x offset), flow[...,1] = v (y offset).
        tile : int
            Tile size for processing. Keeps memory bounded.
        interpolation : {"nearest","linear"}
            Interpolation method.
        border_mode : {"reflect","replicate","constant"}
            Border handling.
        border_value : scalar
            Used if border_mode="constant".
        dtype_out : np.dtype or None
            Output dtype. If None, keep same as input.
        """
        H, W = flow.shape[:2]
        if image.ndim == 2:
            C = 1
            image = image[..., None]
        else:
            C = image.shape[2]
        if dtype_out is None:
            dtype_out = image.dtype

        interp_map = {"nearest": cv.INTER_NEAREST, "linear": cv.INTER_LINEAR}
        border_map = {
            "reflect": cv.BORDER_REFLECT_101,
            "replicate": cv.BORDER_REPLICATE,
            "constant": cv.BORDER_CONSTANT,
        }
        interp_flag = interp_map[interpolation]
        border_flag = border_map[border_mode]

        # Preallocate full output
        out = np.empty((H, W, C), dtype_out)

        for y0 in range(0, H, tile):
            y1 = min(H, y0 + tile)
            for x0 in range(0, W, tile):
                x1 = min(W, x0 + tile)

                h = y1 - y0
                w = x1 - x0

                # Preallocate local buffers
                mapx = np.empty((h, w), dtype=dtype_out)
                mapy = np.empty((h, w), dtype=dtype_out)

                # Base grid
                grid_x, grid_y = np.meshgrid(
                    np.arange(x0, x1, dtype=dtype_out),
                    np.arange(y0, y1, dtype=dtype_out)
                )

                # Fill maps
                mapx[...] = grid_x + flow[y0:y1, x0:x1, 0].astype(dtype_out)
                mapy[...] = grid_y + flow[y0:y1, x0:x1, 1].astype(dtype_out)

                # Sanitize NaN/inf
                bad = ~np.isfinite(mapx) | ~np.isfinite(mapy)
                if np.any(bad):
                    mapx[bad] = -1
                    mapy[bad] = -1

                # Preallocate warped tile buffer
                warped_tile = np.empty((h, w, C), dtype=dtype_out)

                # compute needed source crop
                min_x = max(0, int(np.floor(mapx.min())) - 1)
                max_x = min(W, int(np.ceil(mapx.max())) + 1)
                min_y = max(0, int(np.floor(mapy.min())) - 1)
                max_y = min(H, int(np.ceil(mapy.max())) + 1)
                # slice source
                src_crop = image[min_y:max_y, min_x:max_x, :]
                # adjust maps into local crop coords
                mapx_local = (mapx - min_x).astype(dtype_out)
                mapy_local = (mapy - min_y).astype(dtype_out)
                # Warp each channel into tile buffer, then copy into output
                warped_tile = cv.remap(
                    src_crop,
                    mapx_local, mapy_local,
                    interpolation=interp_flag,
                    borderMode=border_flag,
                    borderValue=border_value
                ).astype(dtype_out, copy=False)

                out[y0:y1, x0:x1, :] = warped_tile

        if C == 1:
            out = out[..., 0]
        return out

    def warp_image(self, image, flow):

        h, w = flow.shape[:2]

        # Create mesh grid of pixel indices
        y, x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')

        # Compute new pixel positions
        x_new = np.clip(x + flow[..., 0], 0, w - 1)
        y_new = np.clip(y + flow[..., 1], 0, h - 1)

        # Warp image using remap
        warped = cv.remap(copy.copy(image), x_new.astype(np.float32), y_new.astype(np.float32),
                          interpolation=cv.INTER_CUBIC, borderMode=cv.BORDER_REPLICATE)
        return warped

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


