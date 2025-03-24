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
        :param images: list of two images
        :return:
        """

        # Reshape the images into correct input, this is for custom batch sizeing
        # tensor_list = [torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) for image in images]
        # stacked_tensor = torch.stack(tensor_list, dim=0)  # Shape: (2, 3, H, W)
        # final_tensor = stacked_tensor.unsqueeze(0)  # Shape: (1, 2, 3, H, W)

        # This is basic function for converting to input of batchsize 1
        input_i = self.io_adapter.prepare_inputs(images)
        return input_i

    def flow(self, input):
        # Forward the inputs through the model
        predictions = self.model(input)
        return predictions

    def torch_flow_to_img(self, flows):
        flow_img = flow_utils.flow_to_rgb(flows)
        flow_rgb = flow_img[0, 0].permute(1, 2, 0)
        flow_rgb_npy = flow_rgb.detach().cpu().numpy()
        # OpenCV uses BGR format
        flow_bgr_npy = cv.cvtColor(flow_rgb_npy, cv.COLOR_RGB2BGR)

        return flow_bgr_npy

    def np_flow_to_img(self, flows):
        flow_img = flow_utils.flow_to_rgb(flows)
        # OpenCV uses BGR format
        flow_bgr_npy = cv.cvtColor(flow_img, cv.COLOR_RGB2BGR)

        return flow_bgr_npy

    def estimate_patches(self, patches, p_key):
        # For saving resulting flows
        flow_patches = []
        idx = 0
        # Save directory for patch visualization
        if self.config.debug:
            debug_save = f"./plots/patches_{p_key}"
            if not os.path.exists(debug_save):
                os.mkdir(debug_save)
        else:
            debug_save = "./plots"

        # Progress bar
        patch_pbar = tqdm(total=len(patches), desc='Processing patches',
                          position=1, leave=False, ncols=100, colour='red')

        for patch in patches:
            idx += 1
            # We reshape so that the estimation is from fragment to overview
            reshaped_patch = (patch[0], patch[1])
            # Check if both images have same shape
            assert patch[0].shape[:2] == patch[1].shape[:2]

            _, bin_mask_p = cv.threshold(cv.cvtColor(patch[0], cv.COLOR_BGR2GRAY), 1, 255, cv.THRESH_BINARY)

            # Get input images
            input_i = self.get_input(reshaped_patch)
            opt_flow = self.flow(input_i)
            # Extract only flow information
            flows = opt_flow['flows']

            if self.config.debug:
                flow_bgr = self.torch_flow_to_img(flows)
                # cv.imwrite(f"{debug_save}/flow{p_key}_{idx}.jpg", (flow_bgr * 255).astype('uint8'))

            # Get the flows into usable format
            flow = flows[0, 0].permute(1, 2, 0).detach().cpu().numpy()
            # Remove flow from non image pixels
            mask = np.expand_dims(bin_mask_p > 0.5, axis=-1)
            flow = flow * mask.astype(flow.dtype)
            # resized_flow = self.optical.upscale_flow(flow, frag.shape[:2])
            warped_flow_fr = self.warp_image(patch[1], flow)
            # Append later for return
            flow_patches.append(flow)
            patch_pbar.update(1)
            if self.config.debug:
                alpha = 0.5
                stitched = (patch[1] * alpha + patch[0] * (1 - alpha))
                #cv.imwrite(f"{debug_save}/unwarped_input_{p_key}_{idx}.jpg", stitched)

                # Stitch flow warped img with overlay
                alpha = 0.5
                stitched = patch[0].copy()
                # stitched[mask] = (stitched[mask] * alpha + warped_flow_fr[mask] * (1 - alpha))
                stitched = (stitched * alpha + warped_flow_fr * (1 - alpha))

                # cv.imwrite(f"{debug_save}/warped_flow{p_key}_{idx}.jpg", warped_flow_fr)

                # cv.imwrite(f"{debug_save}/flow_stitched_{p_key}_{idx}.jpg", stitched)

        return flow_patches

    def estimate_flows(self, warped_images ):
        """
        Estimates the optical flows
        :param warped_images: list of input images with the first being the overlay
        :return: list of optical flows
        """
        # Holders for overlap image, to which every single one is aligned
        # TODO Redo to look nicer
        ov = warped_images.pop(0)
        ov_img = ov[0]
        # ov_mask = ov[0]
        fragment_flows = []

        img_pbar = tqdm(total=len(warped_images), desc='Processing fragments using optical flow',
                        position=0, leave=False, ncols=100, colour='green')
        # Iterate over fragments to compute optical flow in regard to the overlay
        for key, val in warped_images.items():
            # When debugging on single fragment
            # if self.config.debug and key not in [13]:
            #     continue
            # Fragment & maks
            frag = val[0]
            # mask = val[1]

            # Get the overlapping region  for optical flow estimation
            overlap1, overlap2 = self.get_overlap_region(ov_img, frag)


            _, bin_mask_ov = cv.threshold(cv.cvtColor(overlap1, cv.COLOR_BGR2GRAY), 1, 255, cv.THRESH_BINARY)
            _, bin_mask_fr = cv.threshold(cv.cvtColor(overlap2, cv.COLOR_BGR2GRAY), 1, 255, cv.THRESH_BINARY)

            coords_ov = cv.findNonZero(bin_mask_ov)
            coords_fr = cv.findNonZero(bin_mask_fr)

            o_x, o_y, o_w, o_h = cv.boundingRect(coords_ov)
            cropped_ov = overlap1[o_y:o_y + o_h, o_x:o_x + o_w]
            f_x, f_y, f_w, f_h = cv.boundingRect(coords_fr)
            cropped_fr = overlap2[f_y:f_y + f_h, f_x:f_x + f_w]

            if self.config.debug:
                cv.imwrite(f"./plots/cropped_{key}_0.jpg", cropped_ov)
                cv.imwrite(f"./plots/cropped_{key}_1.jpg", cropped_fr)

            # Resize the image into computable form for optical flow estimation
            # useful when subsampling is suitable to predict lower amount of fragment patches
            if self.config.optical.subsample_factor:
                h_f, w_f = self.config.optical.subsample_factor
                h, w = cropped_fr.shape[:2]
                size = (int(h / h_f), int(w / w_f))
                resized_imgs = [cv.resize(cropped_ov, size), cv.resize(cropped_fr, size)]
            else:
                resized_imgs = [cropped_ov, cropped_fr]

            if self.config.debug:
                cv.imwrite(f"./plots/resized_{key}_0.jpg", resized_imgs[0])
                cv.imwrite(f"./plots/resized_{key}_1.jpg", resized_imgs[1])

            # Get the size of patch | [height, width]
            patch_size = self.config.optical.input_size
            # Returns patches list of tuples [(overview, fragment)] patches and relative position of the patch
            patches, positions = self.split_image_with_overlap(resized_imgs, patch_size, self.config.optical.patch_overlap)

            # Estimate the flow on patches
            flow_patches = self.estimate_patches(patches, key)
            # Merge the flow into one whole fragment_flow
            # stitched_flow = self.merge_patches_with_blending(flow_patches,
            #                                                 positions,
            #                                                 resized_imgs[0].shape,
            #                                                 patch_size,
            #                                                 (0, 0))

            stitched_flow = self.merge_flows(flow_patches, positions, resized_imgs[0].shape, self.config.optical.patch_overlap)
            flow_in_ov = np.zeros((ov_img.shape[0],ov_img.shape[1],2))
            # Get the flow into dimension and position of overlap image
            flow_in_ov[o_y:o_y + o_h, o_x:o_x + o_w] = stitched_flow




            if self.config.debug:
                stitched_flow_img = self.np_flow_to_img(stitched_flow)
                cv.imwrite(f"./plots/stitched_flow_{key}.jpg", stitched_flow_img)
                flow_in_ov_img = self.np_flow_to_img(flow_in_ov)
                cv.imwrite(f"./plots/flow_in_ov_img_{key}.jpg", flow_in_ov_img)
                # alpha = 0.5
                # stitched = stiched_patches.copy()
                # stitched = (resized_imgs[0] * alpha + stitched * (1 - alpha))
                # cv.imwrite(f"./plots/stitched_compare_{key}.jpg", stitched)

            fragment_flows.append(flow_in_ov)
            img_pbar.update(1)

        return fragment_flows

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




    def pad_array(self, arr, target_shape, pad_value=0):
        """
        USed for making all patches the same size. If patch size is smaller takes previous pixels to fill it
        :param arr:
        :param target_shape:
        :param pad_value:
        :return:
        """

        h, w = arr.shape[:2]
        th, tw = target_shape[:2]

        pad_h = max(th - h, 0)
        pad_w = max(tw - w, 0)

        pad_width = ((0, pad_h), (0, pad_w)) + ((0, 0),) * (arr.ndim - 2)

        return np.pad(arr, pad_width, mode="constant", constant_values=pad_value)

    def merge_patches_with_blending(self, patches, positions, original_shape, patch_size, overlap):
        h, w, c = original_shape
        o_h, o_w = overlap
        merged_image = np.zeros((h, w, c), dtype=np.uint8)

        for patch, (y, x) in zip(patches, positions):
            ph, pw = patch.shape[:2]

            cut_y = min(y + ph - o_h, h)
            cut_x = min(x + pw - o_w, w)

            start_y = max(0, y - o_h)
            start_x = max(0, x - o_w)

            merged_image[y:cut_y, x:cut_x] += patch[start_y:cut_y - y, start_x:cut_x - x]

        return merged_image

    def warp_image(self, image, flow):
        """Warp an image using an optical flow field."""
        h, w = flow.shape[:2]

        # Create mesh grid of pixel indices
        y, x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')

        # Compute new pixel positions
        x_new = np.clip(x + flow[..., 0], 0, w - 1)
        y_new = np.clip(y + flow[..., 1], 0, h - 1)

        # Pls work
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
