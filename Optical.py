import copy
import torch
import ptlflow
from ptlflow.utils import flow_utils
from ptlflow.utils.io_adapter import IOAdapter
import cv2 as cv
import os
import numpy as np
from tqdm import tqdm


class OpticalFlow:

    def __init__(self, config):
        self.config = config
        self.model = ptlflow.get_model(config.optical.model, config.optical.checkpoint)
        self.model.training = False
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.io_adapter = IOAdapter(self.model, config.optical.input_size, cuda=True)

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
                          position=1, leave=False, ncols=100, colour='red')
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

    def estimate_flows(self, warped_images):
        """
        Estimates the optical flows
        Args:
            warped_images: A dict of shpae {key:(image, mask)} of warped images..

        Returns:

        """
        # Holders for overlap image, to which every single one is aligned
        # TODO Redo to look nicer
        ov = warped_images.pop(0)
        ov_img = ov[0]
        ov_mask = ov[1]
        fragment_flows = []

        # Initialize progress bar
        img_pbar = tqdm(total=len(warped_images), desc='Processing fragments using optical flow',
                        position=0, leave=False, ncols=100, colour='green')

        # Iterate over fragments to compute optical flow in regard to the reference
        for key, val in warped_images.items():

            # Unpack values
            frag = val[0]
            mask = val[1]

            overlap1, overlap2 = self.get_overlap_region(frag, mask[...,0], ov_img, ov_mask[...,0])
            coords_fr, coords_ov = self.get_coords_from_mask(mask, ov_mask)
            # Find bounding rect for both images
            o_x, o_y, o_w, o_h = cv.boundingRect(coords_ov)
            cropped_ov = overlap1[o_y:o_y + o_h, o_x:o_x + o_w]
            f_x, f_y, f_w, f_h = cv.boundingRect(coords_fr)
            cropped_fr = overlap2[f_y:f_y + f_h, f_x:f_x + f_w]

            # Resize the image into computable form for optical flow estimation
            # useful when subsampling is suitable to predict lower amount of fragment patches
            if self.config.optical.subsample_factor:
                h_f, w_f = self.config.optical.subsample_factor
                h, w = cropped_fr.shape[:2]
                size = (int(h / h_f), int(w / w_f))
                resized_imgs = [cv.resize(cropped_ov, size), cv.resize(cropped_fr, size)]
            else:
                resized_imgs = [cropped_ov, cropped_fr]

            # Get the size of patch | [height, width]
            patch_size = self.config.optical.input_size
            # Returns patches, list of tuples [(overview, fragment)] patches and relative position of the patch
            patches, positions = self.split_image_with_overlap(resized_imgs, patch_size, self.config.optical.patch_overlap)
            # Estimate the flow on patches
            flow_patches = self.estimate_patches(patches)

            # Merge the patch flows into whole image
            stitched_flow = self.merge_flows(flow_patches, positions, resized_imgs[0].shape, self.config.optical.patch_overlap)

            # Put the flow into coordinate frame of reference or whole image
            flow_in_ov = np.zeros((ov_img.shape[0], ov_img.shape[1], 2))
            flow_in_ov[o_y:o_y + o_h, o_x:o_x + o_w] = stitched_flow

            # Store the resulting fragment
            fragment_flows.append(flow_in_ov)
            img_pbar.update(1)

            # Print the debug flows
            if self.config.optical.debug:
                cv.imwrite(f"./plots/resized_{key}_0.jpg", resized_imgs[0])
                cv.imwrite(f"./plots/resized_{key}_1.jpg", resized_imgs[1])
                stitched_flow_img = self.np_flow_to_img(stitched_flow)
                cv.imwrite(f"./plots/stitched_flow_{key}.jpg", stitched_flow_img)
                flow_in_ov_img = self.np_flow_to_img(flow_in_ov)
                cv.imwrite(f"./plots/flow_in_ov_img_{key}.jpg", flow_in_ov_img)

        return fragment_flows

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

    def get_overlap_region(self, img_a, mask_a, img_b, mask_b):

        # Find the overlapping region
        overlap_mask = mask_a & mask_b

        # Extract the overlapping areas
        overlap1 = cv.bitwise_and(img_a, img_a, mask=overlap_mask.astype(np.uint8))
        overlap2 = cv.bitwise_and(img_b, img_b, mask=overlap_mask.astype(np.uint8))

        return overlap1, overlap2
