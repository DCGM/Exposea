import cv2 as cv
import numpy
import numpy as np
import torch
import tqdm
import wandb as wb
import torch.nn.functional as F

def equalize(imgs):

    # Get reference image
    ref_img = imgs[0][0]
    adjusted = {}
    # Iterate over all images (skip ref) and equalize light
    for key, val in imgs.items():
        # Ignore reference
        if key == 0:
            continue
        # Unpack variables
        img = val[0]
        mask = val[1]

        # Light optimization
        norm_frag = img.astype(np.float32) / 255.0
        ref_norm = ref_img.astype(np.float32) / 255.0

        frag_adj = spatial_light_adjustment(norm_frag, ref_norm, mask, grid_size=32, mode="scale")

        frag_adj = numpy.asarray(frag_adj * 255.0, dtype=numpy.uint8)

        adjusted[key] = (frag_adj, mask)
        # cv.imwrite(f"./plots/light_{key}_0.jpg", frag_adj)
        # cv.imwrite(f"./plots/light_{key}_1.jpg", adjusted[key][0])
    return adjusted



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


def spatial_light_adjustment(fragment, reference, mask, grid_size=16, mode="scale", num_iters=100, lr=0.1):
    """
       Adjust the lighting of a fragment image to match a reference image with spatially varying correction.

       Args:
           fragment (torch.Tensor): Fragment image (H, W) or (C, H, W), normalized to [0, 1].
           reference (torch.Tensor): Reference image (H, W) or (C, H, W), normalized to [0, 1].
           mask (torch.Tensor, optional): Binary mask (H, W) indicating valid overlap region.
           grid_size (int): Resolution of the low-frequency correction grid.
           mode (str): "scale" for gamma correction or "affine" for alpha-beta correction.
           num_iters (int): Number of optimization iterations.
           lr (float): Learning rate for optimization.

       Returns:
           torch.Tensor: Adjusted fragment image.
       """
    H, W = fragment.shape[:2]
    grid_H, grid_W = grid_size, grid_size
    device = torch.cuda.current_device()

    # Define learnable correction maps (coarse grid values)
    kernel_size = 5
    smooth_kernel = torch.ones(1, 1, kernel_size, kernel_size).to(device) / (kernel_size * kernel_size)

    if mode == "scale":
        # Initialize
        gamma_map = torch.nn.Parameter(torch.randn((1, 1, grid_H, grid_W), device=device, requires_grad=True))
        # gamma_map = torch.nn.Parameter(torch.ones((1, 1, grid_H, grid_W), device=device, requires_grad=True))
        # gamma_map += 0.1 * torch.rand_like(gamma_map)
        #gamma_map = torch.conv2d(gamma_map, smooth_kernel, padding=kernel_size//2)
        gamma_map.requires_grad = True
        params = [gamma_map]
    elif mode == "affine":
        alpha_map = torch.nn.Parameter(torch.ones((1, 1, grid_H, grid_W), device=device, requires_grad=True))
        beta_map = torch.nn.Parameter(torch.ones((1, 1, grid_H, grid_W), device=device, requires_grad=True))
        alpha_map.requires_grad = True
        beta_map.requires_grad = True
        params = [alpha_map, beta_map]
    else:
        raise ValueError("Invalid mode. Use 'scale' or 'affine'.")
    # Optimizer
    optimizer = torch.optim.LBFGS(params, lr=lr, max_iter=num_iters)
    loss_fn = torch.nn.MSELoss()

    ref, frag = reshape_to_lbfgs(reference, fragment, device)

    mask_reshaped =  torch.from_numpy(mask.transpose((2, 0, 1))).float().unsqueeze(0).to(device)

    pbar = tqdm.tqdm(total=num_iters)

    def closure():

        optimizer.zero_grad()

        # Upsample the correction map to full resolution
        if mode == "scale":
            gamma = torch.nn.functional.interpolate(gamma_map, size=(H, W), mode='bilinear', align_corners=True)
            adjusted_fragment = gamma * frag
        else:
            alpha = torch.nn.functional.interpolate(alpha_map, size=(H, W), mode='bilinear', align_corners=True)
            beta = torch.nn.functional.interpolate(beta_map, size=(H, W), mode='bilinear', align_corners=True)
            adjusted_fragment = alpha * frag + beta

        # Compute loss in masked region
        loss = loss_fn(adjusted_fragment.masked_select(mask_reshaped>0), ref.masked_select(mask_reshaped>0))
        loss.backward()

        pbar.update(1)

        return loss

    optimizer.step(closure)

    if mode == "scale":
        gamma = F.interpolate(gamma_map, size=(H, W), mode='bilinear', align_corners=True)
        adjusted_frag = frag * gamma.detach()
    elif mode == "affine":
        alpha = F.interpolate(alpha_map, size=(H, W), mode='bilinear', align_corners=True)
        beta = F.interpolate(beta_map, size=(H, W), mode='bilinear', align_corners=True)
        adjusted_frag = frag * alpha.detach() + beta.detach()

    adjusted_frag = adjusted_frag.detach().clamp(0, 1).cpu()
    adjusted_frag = adjusted_frag[0].numpy().transpose((1, 2, 0))

    return adjusted_frag

#
# def remove_illumination_variations(image, cutoff_frequency=30):
#     """
#     Remove illumination differences in the Value channel using a high-pass frequency filter.
#
#     Parameters:
#     - image: Input RGB image
#     - cutoff_frequency: Radius of the high-pass filter (larger values preserve more low frequencies)
#     """
#     # Convert to HSV
#     hsv_img = cv.cvtColor(image, cv.COLOR_BGR2HSV)
#     h, s, v = cv.split(hsv_img)
#
#     # Step 1: Compute the FFT of the Value channel
#     f = np.fft.fft2(v)
#     fshift = np.fft.fftshift(f)  # Shift zero frequency to the center
#
#     # Step 2: Create a high-pass filter mask
#     rows, cols = v.shape
#     crow, ccol = rows // 2, cols // 2  # Center point
#     mask = np.ones((rows, cols), np.uint8)
#
#     # Define a circular region to block low frequencies
#     cv.circle(mask, (ccol, crow), cutoff_frequency, 0, -1)  # 0 in the center (low frequencies)
#
#     # Step 3: Apply the filter
#     fshift_filtered = fshift * mask
#
#     # Step 4: Inverse FFT
#     f_ishift = np.fft.ifftshift(fshift_filtered)
#     v_filtered = np.fft.ifft2(f_ishift)
#     v_filtered = np.abs(v_filtered)  # Take the magnitude (real part)
#     v_filtered = np.clip(v_filtered, 0, 255).astype(np.uint8)  # Ensure valid pixel values
#
#     # Step 5: Recombine with original H and S channels
#     hsv_filtered = cv.merge([h, s, v_filtered])
#     result = cv.cvtColor(hsv_filtered, cv.COLOR_HSV2BGR)
#
#     return result
#
# def clahe(frag_img):
#
#     hsv = cv.cvtColor(frag_img, cv.COLOR_BGR2HSV)
#     h, s, v = cv.split(hsv)
#     # Apply CLAHE to Value channel
#     clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=[32,32])
#     v_clahe = clahe.apply(v)
#     # Merge and convert back to RGB
#     hsv_clahe = cv.merge([h, s, v_clahe])
#     bgr = cv.cvtColor(hsv_clahe, cv.COLOR_HSV2BGR)
#
#     return bgr
#
#
# def mean_brightness_adj(frag_img, ref_img):
#
#     ref_brightness = np.mean(cv.cvtColor(ref_img, cv.COLOR_BGR2GRAY))
#     frag_brightness = np.mean(cv.cvtColor(frag_img, cv.COLOR_BGR2GRAY))
#     brightness_diff = ref_brightness - frag_brightness
#
#     adjusted_img = cv.convertScaleAbs(frag_img, alpha=1.0, beta=brightness_diff)
#     return adjusted_img
#
#
# def histogram_matching_color(source, reference):
#     # Split into channels
#     mask = source[1]
#     source_channels = cv.split(source[0])
#     ref_channels = cv.split(reference)
#     matched_channels = []
#
#     # Apply histogram matching to each channel
#     for src_chan, ref_chan in zip(source_channels, ref_channels):
#         matched_chan = histogram_matching(src_chan, ref_chan, mask)
#         matched_channels.append(matched_chan)
#
#     # Merge channels back
#     return cv.merge(matched_channels)
#
# # Function to perform histogram matching
# def histogram_matching(frag_img, ref_img, mask):
#     """
#     Adjust the pixel values of the source image to match the histogram of the reference image.
#     """
#     # # Convert images to grayscale if they aren't already (for simplicity)
#     # gr_frag = cv.cvtColor(frag_img_mask[0], cv.COLOR_BGR2GRAY)
#     # gr_ref = cv.cvtColor(ref_img, cv.COLOR_BGR2GRAY)
#     gr_frag = frag_img
#     gr_ref = ref_img
#
#     frag_masked = gr_frag[mask[:,:,0]]
#     ref_masked = gr_ref[mask[:,:,0]]
#
#
#     # Calculate histograms and CDFs
#     source_hist, bins = np.histogram(frag_masked, 256, [0, 256])
#     ref_hist, bins = np.histogram(ref_masked, 256, [0, 256])
#
#     source_cdf = source_hist.cumsum()
#     source_cdf = source_cdf / source_cdf[-1]  # Normalize
#     ref_cdf = ref_hist.cumsum()
#     ref_cdf = ref_cdf / ref_cdf[-1]  # Normalize
#
#     # Create mapping from source to reference
#     mapping = np.zeros(256, dtype=np.uint8)
#     for i in range(256):
#         # Find the closest matching value in the reference CDF
#         j = np.argmin(np.abs(ref_cdf - source_cdf[i]))
#         mapping[i] = j
#
#     # Apply the mapping to the source image
#     matched = cv.LUT(frag_img, mapping)
#     return matched

