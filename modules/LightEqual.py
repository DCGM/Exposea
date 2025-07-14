import cv2 as cv
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from torch import nn
import logging
import sys

class SpatialLightParams(nn.Module):
    def __init__(self, grid_size=32, mode=['scale']):
        super(SpatialLightParams, self).__init__()
        self.grid_H, self.grid_W = grid_size, grid_size
        self.mode = list(mode)


        mode = list(mode)
        if "scale" in mode:
            self.alpha_map = nn.Parameter(torch.ones((1, 1, self.grid_H, self.grid_W)))
            mode.remove("scale")
        elif 'color_scale' in mode:
            self.alpha_map = nn.Parameter(torch.ones((1, 3, self.grid_H, self.grid_W)))
            mode.remove("color_scale")
        if "bias" in mode:
            self.beta_map = nn.Parameter(torch.zeros((1, 1, self.grid_H, self.grid_W)))
            mode.remove("bias")

        # Gamma does not work
        if 'gamma' in mode:
            self.gamma1 = nn.Parameter(torch.zeros((1, 1, 1, 1)))
            self.gamma2 = nn.Parameter(torch.zeros((1, 1, 1, 1)))
            mode.remove("gamma")
        if mode:
            raise ValueError(f"Unknown modes: {mode}")

    def interpolate(self, img):
        H, W = img.shape[2:4]

        adjusted_fragment = img

        # First apply gamma - this will be inverted at the end with 1/gamma
        if 'gamma' in self.mode:
            gamma = 1.2**self.gamma1
            adjusted_fragment = adjusted_fragment.pow(gamma)

        if 'scale' in self.mode or 'color_scale' in self.mode:
            alpha = torch.nn.functional.interpolate(self.alpha_map, size=(H, W), mode='bilinear', align_corners=True)
            adjusted_fragment = alpha * adjusted_fragment

        if 'bias' in self.mode:
            beta = torch.nn.functional.interpolate(self.beta_map, size=(H, W), mode='bilinear', align_corners=True)
            adjusted_fragment = adjusted_fragment + beta

        if 'gamma' in self.mode:
            # clip values to avoid overflow
            adjusted_fragment = adjusted_fragment.clamp(1.0e-9, 1)
            gamma = 1.2 ** self.gamma2
            adjusted_fragment = adjusted_fragment.pow(1/gamma)

        return adjusted_fragment


def split_image(img, fragment_height, fragment_width):
    img_h, img_w, _ = img.shape
    fragments = []
    for y in range(0, img_h, fragment_height):
        for x in range(0, img_w, fragment_width):
            frag = img[y:y+fragment_height, x:x+fragment_width]
            fragments.append(((y, x), frag))
    return fragments

def compose_image(fragments, full_shape):
    result = np.zeros(full_shape, dtype=np.float32)
    for (y, x), frag in fragments:
        h, w = frag.shape[:2]
        result[y:y+h, x:x+w] = frag
    return result

def tile_equalize_fragments(flow_fragment, mask, ref_img, config):
    # Get reference image and normalize it
    ref_norm = ref_img.astype(np.float32) / 255.0
    # Normalize fragment
    norm_frag = flow_fragment.astype(np.float32) / 255.0
    # Cut out only the area of fragment not the whole final res
    y_min, x_min = np.argwhere(mask[:, :, 0]).min(axis=0)  # Get min row and column
    y_max, x_max = np.argwhere(mask[:, :, 0]).max(axis=0)  # Get max row and column
    cut_frag = norm_frag[y_min:y_max, x_min:x_max]
    cut_ref = ref_norm[y_min:y_max, x_min:x_max]
    mask_cut = mask[y_min:y_max, x_min:x_max]

    tile_size = config.light_optim.tile_size
    tile_frag = split_image(cut_frag, tile_size[1], tile_size[0])
    tile_ref = split_image(cut_ref, tile_size[1], tile_size[0])
    tile_mask = split_image(mask_cut, tile_size[1], tile_size[0])

    adjusted_frags = []
    for f, r, m in zip(tile_frag, tile_ref, tile_mask):
        adjusted, _ = spatial_light_adjustment(f[1], r[1], m[1], config)
        adjusted_frags.append((f[0],adjusted))

    composed = compose_image(adjusted_frags, cut_frag.shape)
    frag_adj = np.zeros_like(flow_fragment, dtype=np.float32)
    frag_adj[y_min:y_max, x_min:x_max] = composed   # Rescale it back to 255
    frag_adj = np.asarray(frag_adj * 255.0, dtype=np.uint8)
    # cv.imwrite("../plots/composed.jpg", frag_adj)
    return frag_adj, None


def equalize_frag(flow_fragment, mask, ref_img, config):
    # Get reference image and normalize it
    ref_norm = ref_img.astype(np.float32) / 255.0
    # Normalize fragment
    norm_frag = flow_fragment.astype(np.float32) / 255.0
    # Cut out only the area of fragment not the whole final res
    y_min, x_min = np.argwhere(mask[:, :, 0]).min(axis=0)  # Get min row and column
    y_max, x_max = np.argwhere(mask[:, :, 0]).max(axis=0)  # Get max row and column
    cut_frag = norm_frag[y_min:y_max, x_min:x_max]
    cut_ref = ref_norm[y_min:y_max, x_min:x_max]
    mask_cut = mask[y_min:y_max, x_min:x_max]

    if config.debug:
        logging.info(f"Equalizing fragment size: {cut_frag.shape}")

    # Light optimization
    frag_cut, m = spatial_light_adjustment(cut_frag, cut_ref, mask_cut, config)
    frag_adj = np.zeros_like(norm_frag)
    frag_adj[y_min:y_max, x_min:x_max] = frag_cut
    # Rescale it back to 255
    frag_adj = np.asarray(frag_adj * 255.0, dtype=np.uint8)

    return frag_adj, m


def spatial_light_adjustment(fragment, reference, mask, config):
    """
        Adjust the lighting of a fragment image to match a reference image with spatially varying correction.
        :param
           fragment: Fragment image (H, W, C) normalized to [0, 1].
           reference: Reference image (H, W, C) normalized to [0, 1].
           mask : Binary mask (H, W, C) indicating valid overlap region.
           config: Configuration object containing all configuration parameters.

       :return
           adjusted: CV Image with adjusted lighting correction.
       """

    device = torch.cuda.current_device()

    method = SpatialLightParams(grid_size=config.stitcher.grid_size, mode=config.stitcher.optim_mode)
    method.to(device)

    loss = config.stitcher.loss_type
    if loss == "mse":
        loss_fn = torch.nn.MSELoss()
    elif loss == "l1":
        loss_fn = torch.nn.L1Loss()
    elif loss == "gauss_smooth":
        loss_fn = gauss_smooth_loss
    elif loss == "l1_smooth":
        loss_fn = l1_smooth_loss
    else:
        raise ValueError(f"Unknown loss function: {loss}")


    # Reshapes frag and reference from (H,W,C) -> (B,C,H,W) and put tensor to device
    ref, frag = reshape_to_lbfgs(reference, fragment, device)
    # Reshapes the mask so the loss calculation is easier
    mask_reshaped = torch.from_numpy(mask.transpose((2, 0, 1))).float().unsqueeze(0).to(device)

    with torch.no_grad():
        loss = loss_fn(method.interpolate(frag).masked_select(mask_reshaped>0), ref.masked_select(mask_reshaped>0))
        logging.info(f"Initial loss: {loss.item():.4f}")
        del loss

    if config.stitcher.optimizer == 'adam':
        # Initialize the optimizer
        optimizer = torch.optim.Adam(method.parameters(), lr=config.stitcher.lr)

        # Progress bar
        pbar = tqdm.tqdm(total=config.stitcher.optim_steps, leave=False, ncols=100, colour='green', file=sys.stdout)

        for i in range(config.stitcher.optim_steps):
            optimizer.zero_grad()
            # Upsample the correction map to full resolution
            adjusted_fragment = method.interpolate(frag)
            loss = loss_fn(adjusted_fragment.masked_select(mask_reshaped>0), ref.masked_select(mask_reshaped>0))
            loss.backward()
            optimizer.step()
            pbar.set_description(f"Loss: {loss.item():.4f}")
            pbar.update(1)


    if config.stitcher.optimizer == "LBFGS":
        # Initialize the optimizer
        optimizer = torch.optim.LBFGS(method.parameters(),
                                      lr=config.stitcher.lr,
                                      max_iter=config.stitcher.optim_steps,
                                      line_search_fn="strong_wolfe",
                                      tolerance_grad= 1e-12,
                                      tolerance_change=1e-12)

        # Progress bar
        pbar = tqdm.tqdm(total=config.stitcher.optim_steps, leave=False, ncols=100, colour='green', file=sys.stdout)
        def closure():
            optimizer.zero_grad()
            # Upsample the correction map to full resolution
            adjusted_fragment = method.interpolate(frag)
            loss = loss_fn(adjusted_fragment.masked_select(mask_reshaped>0), ref.masked_select(mask_reshaped>0))
            loss.backward()
            pbar.set_description(f"Loss: {loss.item():.4f}")
            pbar.update(1)
            return loss

        optimizer.step(closure)

    # Return final interpolated image
    adjusted_frag = method.interpolate(frag)

    # Adjust the tensor back to cv img representation
    adjusted_frag = adjusted_frag.detach().clamp(0, 1).cpu()
    adjusted_frag = adjusted_frag[0].numpy().transpose((1, 2, 0))
    del frag, ref
    torch.cuda.empty_cache()
    return adjusted_frag, method


def gauss_smooth_loss(predictions, target, params, lambda_smooth=1e-3):
    mse_loss = torch.nn.functional.mse_loss(predictions, target)
    smoothed_params = F.avg_pool1d(params.unsqueeze(0), kernel_size=3, stride=1, padding=1).squeeze(0)
    smoothness_loss = torch.sum((params - smoothed_params)**2)
    return mse_loss + lambda_smooth * smoothness_loss

def l1_smooth_loss(predictions, target, params, lambda_tv=1e-3):
    mse_loss = torch.nn.functional.mse_loss(predictions, target)
    tv_loss = torch.sum((params[1:] - params[:-1])**2)  # L2 smoothness
    return mse_loss + lambda_tv * tv_loss

def reshape_to_lbfgs(reference, fragment, device):
    ref = reference.transpose((2, 0, 1))
    frag = fragment.transpose((2, 0, 1))
    ref =  torch.from_numpy(ref).float().unsqueeze(0).to(device)
    frag = torch.from_numpy(frag).float().unsqueeze(0).to(device)
    return ref, frag