import cv2 as cv
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from torch import nn
import logging

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



def equalize(fragments, frag_cache, ref, config):
    """
    Main function that iterates over all fragments to equalize them with regards to reference
    :param
         imgs dict {name: (image, mask)}: Dictionary contains all images with reference. Every image has its own mask
    :return: dict {name: (image, mask)} where images are adjusted
    """
    # Get reference image and normalize it
    ref_norm = ref.astype(np.float32) / 255.0

    # Iterate over all images (skip ref) and equalize light. Store the resulting image to adjusted dict
    adjusted = []
    for idx, val in enumerate(fragments):
        if frag_cache is not None:
            frag, mask = frag_cache.pop(val)
        else:
            frag, mask = val[0], val[1]
        logging.info(f"Processing image {idx} with mask {mask.shape}")
        # Normalize fragment
        norm_frag = frag.astype(np.float32) / 255.0
        # Light optimization
        frag_adj = spatial_light_adjustment(norm_frag, ref_norm, mask, config)

        # Rescale it back to 255
        frag_adj = np.asarray(frag_adj * 255.0, dtype=np.uint8)
        # Append the result
        if frag_cache is not None:
            frag_key = f"light_adjusted_{idx}"
            frag_cache[frag_key] = (frag_adj, mask)
            adjusted.append(frag_key)
        else:
            adjusted.append((frag_adj, mask))

    return adjusted



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
        pbar = tqdm.tqdm(total=config.stitcher.num_iters)

        for i in range(config.stitcher.num_iters):
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
        pbar = tqdm.tqdm(total=config.stitcher.optim_steps)
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

    return adjusted_frag


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