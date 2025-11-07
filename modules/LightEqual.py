import cv2 as cv
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from torch import nn
import logging
import sys

class SpatialLightParams(nn.Module):
    """
    Implements a neural network module for spatial light parameters adjustment.

    This class is designed to manipulate certain spatial light adjustment parameters such as
    scaling, bias, and gamma transformation. It allows configurable modes that dictate which
    transformations will be applied and provides the ability to interpolate these adjustments
    to match the dimensions of the input images.

    Attributes:
        grid_H (int): Height of the grid used for spatial adjustments.
        grid_W (int): Width of the grid used for spatial adjustments.
        mode (list of str): Modes of operation, defining the types of spatial adjustments
            applied (e.g., "scale", "bias", "color_scale", or "gamma").
        alpha_map (torch.nn.Parameter, optional): A parameter representing the scaling factors
            for spatial adjustment depending on the selected mode.
        beta_map (torch.nn.Parameter, optional): A parameter representing the bias adjustments
            for spatial adjustment depending on the selected mode.
        gamma1 (torch.nn.Parameter, optional): A parameter used in gamma adjustment
            for forward transformation.
        gamma2 (torch.nn.Parameter, optional): A parameter used in gamma adjustment
            for reverse transformation.

    Methods:
        interpolate(img):
            Adjusts the input image based on the configuration of modes and the parameter maps.

    Raises:
        ValueError: If an unknown mode is specified in the `mode` parameter.
    """
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
        """
        Adjusts image fragments by applying various transformations such as gamma correction,
        scaling, color scaling, and biasing. The transformations are determined based on the
        mode and attributes specified in the instance.

        Parameters:
            img: Tensor
                A tensor representing image data. The last two dimensions indicate the height
                and width of the image.

        Returns:
            Tensor: A tensor representing the adjusted image fragment.
        """
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
    """
    Splits an image into smaller fragments of specified height and width.

    Parameters:
        img: The input image as a 3D NumPy array.
        fragment_height: int
            The height of each fragment in pixels.
        fragment_width: int
            The width of each fragment in pixels.

    Returns:
        list[tuple[tuple[int, int], numpy.ndarray]]: A list of tuples where each
        tuple contains:
            - Coordinates (tuple of two integers): The top-left corner of the
              fragment relative to the original image.
            - Fragment (NumPy array): The extracted fragment of the image.
    """
    img_h, img_w, _ = img.shape
    fragments = []
    for y in range(0, img_h, fragment_height):
        for x in range(0, img_w, fragment_width):
            frag = img[y:y+fragment_height, x:x+fragment_width]
            fragments.append(((y, x), frag))
    return fragments

def compose_image(fragments, full_shape):
    """
    Composes a full image by placing fragments at specified positions.

    Args:
        fragments (list[tuple[tuple[int, int], numpy.ndarray]]): A list of
            tuples where each tuple contains a position as (y, x) and the
            corresponding image fragment as a NumPy array.
        full_shape (tuple[int, int]): Shape of the full image to be composed
            as (height, width).

    Returns:
        numpy.ndarray: The composed full image with all fragments placed at
        their specified positions.
    """
    result = np.zeros(full_shape, dtype=np.float32)
    for (y, x), frag in fragments:
        h, w = frag.shape[:2]
        result[y:y+h, x:x+w] = frag
    return result

def tile_equalize_fragments(flow_fragment, mask, ref_img, config):
    """
    Normalizes a fragmented image and adjusts its lighting by dividing it into smaller tiles
    and equalizing their lighting with reference image tiles. The adjusted fragments are
    recomposed into the original dimensions.

    Parameters:
        flow_fragment (np.ndarray): The original fragmented image to be processed.
        mask (np.ndarray): The binary mask outlining the areas in the image to be adjusted.
        ref_img (np.ndarray): The reference image used for lighting adjustments.
        config: Configuration object containing settings such as tile size for adjustments.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: The adjusted image with normalized lighting.
            - None: Placeholder for future extensions.
    """
    logger = logging.getLogger("LIGHT OPTIM")

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
    frag_adj[y_min:y_max, x_min:x_max] = composed
    # Rescale it back to 255
    frag_adj = frag_adj * 255.0
    # cv.imwrite("../plots/composed.jpg", frag_adj)
    return frag_adj, None


def equalize_frag(flow_fragment, mask, ref_img, config):
    """
    Normalizes a fragmented image and equalizing their lighting with reference image.

    Args:
        flow_fragment: The flow fragment represented as a numpy array, to be light-adjusted.
        mask: Mask representing the region of interest in the flow fragment.
        ref_img: Reference image represented as a numpy array, used for light adjustment.
        config: Configuration object containing settings for various operations. Must include a 'debug'
            attribute.

    Returns:
        A tuple containing:
            - Adjusted flow fragment with light settings equalized to the reference image.
            - Metadata or parameters 'm' resulting from the spatial light adjustment process.

    Raises:
        ValueError: If the required dimensions, data types, or configurations are not met.

    """
    logger = logging.getLogger("LIGHT OPTIM")
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
        logger.info(f"Equalizing fragment size: {cut_frag.shape}")

    # Light optimization
    frag_cut, m = spatial_light_adjustment(cut_frag, cut_ref, mask_cut, config)
    frag_adj = np.zeros_like(norm_frag)
    frag_adj[y_min:y_max, x_min:x_max] = frag_cut
    # Rescale it back to 255
    frag_adj = np.asarray(frag_adj * 255.0, dtype=np.float32)

    return frag_adj, m


def spatial_light_adjustment(fragment, reference, mask, config):
    """
    Optimizes the lighting in a spatially aware manner for an input image fragment based on a reference image,
    using configurable optimization methods.

    This function optimizes the lighting conditions of an image fragment to match those of a reference
    image within a specified mask. It uses different loss functions and optimization techniques
    (adam or LBFGS) based on the provided configuration parameters.

    Arguments:
        fragment (torch.Tensor): The input image fragment that needs adjustment. Shape expected as (H, W, C).
        reference (torch.Tensor): The reference image that defines the target lighting. Shape expected as (H, W, C).
        mask (numpy.ndarray): Mask specifying the region to match the lighting. Shape expected as (H, W, C).
        config: Configuration object containing parameters for the optimization process.

    Returns:
        Tuple[torch.Tensor, SpatialLightParams]: A tuple where the first element is the final adjusted image
                                                 in shape (H, W, C) and the second element is the optimized
                                                 SpatialLightParams model instance.
    """
    logger = logging.getLogger("LIGHT OPTIM")
    device = torch.cuda.current_device()

    method = SpatialLightParams(grid_size=config.light_optim.grid_size, mode=config.light_optim.mode)
    method.to(device)

    loss = config.light_optim.loss_type
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
    final_loss = 0
    with torch.no_grad():
        loss = loss_fn(method.interpolate(frag).masked_select(mask_reshaped>0), ref.masked_select(mask_reshaped>0))
        #logger.info(f"Initial loss: {loss.item():.4f}")
        del loss

    if config.light_optim.optimizer == 'adam':
        # Initialize the optimizer
        optimizer = torch.optim.Adam(method.parameters(), lr=config.light_optim.lr)

        # Progress bar
        pbar = tqdm.tqdm(total=config.light_optim.steps, leave=False, ncols=100, colour='green', file=sys.stdout)

        for i in range(config.light_optim.steps):
            optimizer.zero_grad()
            # Upsample the correction map to full resolution
            adjusted_fragment = method.interpolate(frag)
            loss = loss_fn(adjusted_fragment.masked_select(mask_reshaped>0), ref.masked_select(mask_reshaped>0))
            loss.backward()
            optimizer.step()
            pbar.set_description(f"Loss: {loss.item():.4f}")
            pbar.update(1)
        final_loss = loss.item()

    if config.light_optim.optimizer == "LBFGS":
        # Initialize the optimizer
        optimizer = torch.optim.LBFGS(method.parameters(),
                                      lr=config.light_optim.lr,
                                      max_iter=config.light_optim.steps,
                                      line_search_fn="strong_wolfe",
                                      tolerance_grad= 1e-12,
                                      tolerance_change=1e-12)

        final_loss = [None]
        # Progress bar
        pbar = tqdm.tqdm(total=config.light_optim.steps, leave=False, ncols=100, colour='green', file=sys.stdout)

        def closure():
            optimizer.zero_grad()
            # Upsample the correction map to full resolution
            adjusted_fragment = method.interpolate(frag)
            loss = loss_fn(adjusted_fragment.masked_select(mask_reshaped>0), ref.masked_select(mask_reshaped>0))
            final_loss[0] = loss.item()
            loss.backward()
            pbar.set_description(f"Loss: {loss.item():.4f}")
            pbar.update(1)
            return loss

        optimizer.step(closure)
        final_loss = final_loss[0]

    logger.info(f"Final loss: {final_loss:.4f}")
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