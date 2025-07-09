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
        for patch in patches:
            idx += 1
            # We reshape so that the estimation is from fragment to overview
            reshaped_patch = (patch[0], patch[1])
            # Check if both images have same shape
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
        # with torch.profiler.profile(
        #         activities=[torch.profiler.ProfilerActivity.CUDA],
        #         profile_memory=True,
        #         with_stack=True
        # ) as prof:
        # Get the overlapping region  for optical flow estimation
        overlap1, overlap2 = self.get_overlap_region(ref_img, frag_img)
        assert overlap1.shape == overlap2.shape

        _, bin_mask_ov = cv.threshold(cv.cvtColor(overlap1, cv.COLOR_BGR2GRAY), 1, 255, cv.THRESH_BINARY)
        # _, bin_mask_fr = cv.threshold(cv.cvtColor(overlap2, cv.COLOR_BGR2GRAY), 1, 255, cv.THRESH_BINARY)

        coords_ov = cv.findNonZero(bin_mask_ov)
        # coords_fr = cv.findNonZero(bin_mask_fr)

        o_x, o_y, o_w, o_h = cv.boundingRect(coords_ov)
        cropped_ov = overlap1[o_y:o_y + o_h, o_x:o_x + o_w]
        # f_x, f_y, f_w, f_h = cv.boundingRect(coords_fr)
        # cropped_fr = overlap2[f_y:f_y + f_h, f_x:f_x + f_w]
        cropped_fr = overlap2[o_y:o_y + o_h, o_x:o_x + o_w]

        if self.config.optical.debug:
            cv.imwrite(f"./plots/cropped_ov_{debug_idx}.jpg", cropped_ov)
            cv.imwrite(f"./plots/cropped_fr_{debug_idx}.jpg", cropped_fr)

        if cropped_fr.shape != cropped_ov.shape:
            raise ValueError("Images must have the same shape.")

        # Subsample and blur the fragment for matching the resolutions
        if self.config.optical.adjust:
            resized_imgs = self.adjust_images(cropped_ov, cropped_fr, self.config.optical.adjust_params)
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
        # If subsampled then upsample to original
        if resized_imgs[0].shape != cropped_fr.shape:
            stitched_flow = cv.resize(stitched_flow, (cropped_fr.shape[1], cropped_fr.shape[0]), interpolation=cv.INTER_LINEAR)

        flow_in_ov = np.zeros((ref_img.shape[0], ref_img.shape[1], 2))
        # Get the flow into dimension and position of overlap image
        flow_in_ov[o_y:o_y + o_h, o_x:o_x + o_w] = stitched_flow

        if self.config.debug:
            stitched_flow_img = self.np_flow_to_img(stitched_flow)
            cv.imwrite(f"./plots/stitched_flow_{debug_idx}.jpg", stitched_flow_img)
            flow_in_ov_img = self.np_flow_to_img(flow_in_ov)
            cv.imwrite(f"./plots/flow_in_ov_img_{debug_idx}.jpg", flow_in_ov_img)

        #logging.warning(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))
        return flow_in_ov


    def adjust_images(self, reference, fragment, config):
        # Step 2: Anti-aliasing Gaussian blur before resizing
        sigma = config.gaus_blur_sigma # adjust based on scale
        blurred_frag = cv.GaussianBlur(fragment, (5, 5), sigma)
        if config.debug:
            cv.imwrite(f"./plots/blurred_{self.debug_idx}.jpg", blurred_frag)
            cv.imwrite(f"./plots/unblured_{self.debug_idx}.jpg", fragment)

        if config.subsample_factor:
            height, width = blurred_frag.shape[0],  blurred_frag.shape[1]
            scale = 1 / config.subsample_factor
            new_size = (int(width * scale), int(height * scale))
            ds_frag = cv.resize(blurred_frag, new_size, interpolation=cv.INTER_AREA)
            ds_ref = cv.resize(reference, new_size, interpolation=cv.INTER_AREA)
        else:
            ds_frag = blurred_frag
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

        ph -= 2 * oh
        pw -= 2 * ow

        patches = []
        positions = []
        # TODO Cases when image is the exac
        for y in range(0, h , ph ):
            for x in range(0, w , pw):

                if y + ph > h:
                    oversize_h = (y + ph) - h
                    y = y - oversize_h

                if x + pw > w:
                    oversize_w = (x + pw) - w
                    x = x - oversize_w

                patch_a = image_a[max(0, y - oh) : y + min(ph + oh, h), max(0, x - ow):x + min(pw + ow, w)]
                patch_b = image_b[max(0, y - oh) : y + min(ph + oh, h), max(0, x - ow):x + min(pw + ow, w)]

                patches.append((patch_a, patch_b))
                positions.append((y, x))

        return patches, positions


    def merge_flows(self, flows, positions, original_shape, overlap):
        """
        Merges individual patches to the original image
        Args:
            flows: list of flow patches
            positions: relative positions of patches
            original_shape: original image shape
            overlap: overlap size of patches

        Returns:
            returns the merged fragment
        """
        # unpack some values
        img_h, img_w, _ = original_shape
        oh, ow = overlap
        p_h, p_w = self.config.optical.input_size

        p_h -= 2 * oh
        p_w -= 2 * ow

        merged_flow = np.zeros((img_h, img_w, 2), dtype=np.float32)
        merged_acc = np.zeros((img_h, img_w, 2), dtype=np.float32) + 1e-5
        for idx, (patch, (y, x)) in enumerate(zip(flows, positions)):

            cut_y = min(y + p_h, img_h)
            cut_x = min(x + p_w, img_w)

            if y == 0:
                cut_y += oh
            else:
                patch = patch[oh: p_h + oh, :]
            if x == 0:
                cut_x += ow
            else:
                patch = patch[:, ow :p_w + ow]

            merged_flow[y:cut_y, x:cut_x] += patch[:,:]
            merged_acc[y:cut_y, x:cut_x] += [1, 1]
        # TODO Possible problem here :)
        normalized_flow = merged_flow / merged_acc

        return normalized_flow


    def warp_image(self, image, flow):
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
                          interpolation=cv.INTER_CUBIC)
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


