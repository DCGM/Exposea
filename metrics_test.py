import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = '10000000000'
import os.path as osp
import argparse
import shutil

import pyiqa as iqa

from concurrent.futures import ProcessPoolExecutor, as_completed
import subprocess

import pprint
import pickle

import cv2 as cv
import tifffile
import numpy as np
import torch.cuda
import datetime
from scipy.ndimage import map_coordinates
from scipy.optimize import least_squares
from omegaconf import OmegaConf

from modules.HomogEst import HomogEstimator
from modules.Stitcher import Stitcher, ActualBlender, DebugBlender
from modules.Optical import OpticalFlow
from modules.LightEqual import *
from utils.rectangularize import clip, order_points
from utils.utils import scale_homog

import gc


import torch
import torch.nn.functional as F
import pyiqa as iqa

def align_affine_and_light_torch(
    ref, src,
    mask_ref=None,
    iters=400,
    lr=5e-2,
    rotation_max_deg=15.0,
    shear_max=0.1,
    scale_min=0.75,
    scale_max=1.25,
    robust_charb_eps=1e-3,
    device=None,
):
    """
    Jointly optimize affine alignment + global light (gain/bias) in PyTorch.

    Model:
        warp: affine (src -> ref coords) via grid_sample
        light: per-channel gain/bias: src_adj = a * src_warp + b

    Inputs
    ------
    ref, src : torch.Tensor or numpy array
        Shape (H,W), (H,W,C), (C,H,W), (1,C,H,W) etc. Converted internally to (1,C,H,W).
        Recommended value range: float in [0,1] or [0,255] (either works; optimization is easier if normalized).
    mask_ref : optional mask in ref coords
        Shape (H,W) or broadcastable. 1 = include in loss.
    Returns
    -------
    src_adj : (H,W,C) float tensor on CPU
        Warped + photometrically adjusted source in ref frame.
    valid_mask : (H,W) uint8 tensor on CPU
        1 where warped pixels are valid (inside source image) AND mask_ref (if provided).
    params : dict
        theta (2x3), a (C,), b (C,)
    """

    # --------------------------
    # Helpers: shape to BCHW
    # --------------------------
    def to_bchw(x):
        x = torch.as_tensor(x)
        if x.ndim == 2:              # H W
            x = x[None, None, ...]
        elif x.ndim == 3:
            if x.shape[0] in (1, 3, 4):   # C H W
                x = x[None, ...]
            else:                         # H W C
                x = x.permute(2, 0, 1)[None, ...]
        elif x.ndim == 4:
            pass
        else:
            raise ValueError(f"Unsupported shape: {tuple(x.shape)}")
        return x

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    ref = to_bchw(ref).float().to(device)
    src = to_bchw(src).float().to(device)

    if ref.shape[0] != 1 or src.shape[0] != 1:
        raise ValueError("This helper expects batch size 1 (it returns a single aligned image).")

    _, C, H, W = ref.shape
    if src.shape[1] != C:
        # If you pass grayscale + color etc., force both to grayscale
        if C != 1:
            ref = ref.mean(dim=1, keepdim=True)
            C = 1
        if src.shape[1] != 1:
            src = src.mean(dim=1, keepdim=True)

    # Optional mask in ref coords
    if mask_ref is None:
        mask_ref_t = torch.ones((1, 1, H, W), device=device)
    else:
        mask_ref_t = to_bchw(mask_ref).float().to(device)
        if mask_ref_t.shape[-2:] != (H, W):
            raise ValueError("mask_ref must have same H,W as ref")
        mask_ref_t = (mask_ref_t > 0).float()

    # Normalize (helps optimization)
    # (Keeps relative intensities; also fine to remove if you work in 0..1 already)
    ref_n = (ref - ref.mean(dim=(-2,-1), keepdim=True)) / (ref.std(dim=(-2,-1), keepdim=True) + 1e-6)
    src_n = (src - src.mean(dim=(-2,-1), keepdim=True)) / (src.std(dim=(-2,-1), keepdim=True) + 1e-6)

    # --------------------------
    # Parameterization with bounds
    # --------------------------
    # We parametrize a "near-similarity + small shear" affine:
    # A = R(theta) @ S(sx, sy) @ Sh(sh)
    # translation in normalized coords (tx, ty) in [-1,1]
    #
    # Use unconstrained vars u and map with tanh/sigmoid to bounds.
    rot_max = torch.tensor(rotation_max_deg * 3.1415926535 / 180.0, device=device)

    u_theta = torch.zeros((), device=device, requires_grad=True)
    u_tx    = torch.zeros((), device=device, requires_grad=True)
    u_ty    = torch.zeros((), device=device, requires_grad=True)
    u_sx    = torch.zeros((), device=device, requires_grad=True)
    u_sy    = torch.zeros((), device=device, requires_grad=True)
    u_sh    = torch.zeros((), device=device, requires_grad=True)

    # Per-channel light
    u_a = torch.zeros((C,), device=device, requires_grad=True)   # mapped to positive
    u_b = torch.zeros((C,), device=device, requires_grad=True)   # bias free

    def build_theta():
        theta = torch.tanh(u_theta) * rot_max
        tx = torch.tanh(u_tx)   # normalized translation in [-1,1]
        ty = torch.tanh(u_ty)

        # scale in [scale_min, scale_max]
        sx = scale_min + (scale_max - scale_min) * torch.sigmoid(u_sx)
        sy = scale_min + (scale_max - scale_min) * torch.sigmoid(u_sy)

        # shear in [-shear_max, shear_max]
        sh = torch.tanh(u_sh) * shear_max

        c = torch.cos(theta)
        s = torch.sin(theta)

        # R
        R = torch.stack([
            torch.stack([c, -s]),
            torch.stack([s,  c]),
        ])  # 2x2

        # S
        S = torch.diag(torch.stack([sx, sy]))  # 2x2

        # Shear (x += sh*y)
        Sh = torch.stack([
            torch.stack([torch.tensor(1.0, device=device), sh]),
            torch.stack([torch.tensor(0.0, device=device), torch.tensor(1.0, device=device)]),
        ])  # 2x2

        A = R @ S @ Sh  # 2x2

        # Convert to 2x3 theta for affine_grid: maps ref grid -> src sampling grid
        # Important convention:
        # - affine_grid(theta) produces a grid in source coords (normalized) for each output pixel
        # - grid_sample samples src at those coords to create output aligned to ref
        #
        # So theta should map output (ref) normalized coords to input (src) normalized coords.
        M = torch.zeros((2,3), device=device)
        M[:2,:2] = A
        M[0,2] = tx
        M[1,2] = ty
        return M

    def light_params():
        # gain positive, centered around 1
        a = torch.exp(u_a)  # >0
        b = u_b
        return a, b

    # Precompute a ones mask to get valid overlap from grid_sample
    ones_src = torch.ones((1, 1, src.shape[2], src.shape[3]), device=device)

    opt = torch.optim.Adam([u_theta,u_tx,u_ty,u_sx,u_sy,u_sh,u_a,u_b], lr=lr)

    for _ in range(iters):
        theta = build_theta()[None, ...]  # 1x2x3

        grid = F.affine_grid(theta, size=(1, C, H, W), align_corners=False)

        # Warp src and also warp a ones-mask to detect valid samples
        src_w = F.grid_sample(src_n, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        valid = F.grid_sample(ones_src, grid, mode="nearest", padding_mode="zeros", align_corners=False)

        a, b = light_params()
        a_ = a.view(1, C, 1, 1)
        b_ = b.view(1, C, 1, 1)
        src_w_adj = a_ * src_w + b_

        # Charbonnier robust loss on overlap
        w = (valid > 0.5).float() * mask_ref_t
        diff = ref_n - src_w_adj
        loss_map = torch.sqrt(diff * diff + robust_charb_eps * robust_charb_eps)

        loss = (loss_map * w).sum() / (w.sum() + 1e-6)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    # Final outputs computed on original (non-normalized) src
    with torch.no_grad():
        theta = build_theta()[None, ...]
        grid = F.affine_grid(theta, size=(1, C, H, W), align_corners=False)

        src_w = F.grid_sample(src, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        valid = F.grid_sample(ones_src, grid, mode="nearest", padding_mode="zeros", align_corners=False)

        a, b = light_params()
        src_adj = a.view(1,C,1,1) * src_w + b.view(1,C,1,1)

        w = ((valid > 0.5).float() * mask_ref_t)
        valid_mask = (w[0,0] > 0.5).to(torch.uint8)

        # return HWC on CPU for convenience
        out = src_adj[0].permute(1,2,0).cpu() if C > 1 else src_adj[0,0].cpu()

        params = {
            "theta_2x3": build_theta().detach().cpu(),
            "a": a.detach().cpu(),
            "b": b.detach().cpu(),
        }

    return out, valid_mask.cpu(), params

import torch
import torch.nn.functional as F

def align_affine_with_closedform_light_lbfgs(
    ref, src,
    mask_ref=None,
    iters=80,
    lr=1.0,
    rotation_max_deg=5.0,
    shear_max=0.02,
    scale_min=0.9,
    scale_max=1.1,
    charb_eps=1e-3,
    eps=1e-6,
    device=None,
):
    """
    Align src -> ref with restricted affine. Lighting is solved in closed form each iteration:
        src_adj = a * src_w + b   (a,b per-channel)
    Returns adjusted warped src, valid mask, and params (theta, a, b).
    """

    def to_bchw(x):
        x = torch.as_tensor(x)
        if x.ndim == 2:              # H W
            x = x[None, None]
        elif x.ndim == 3:
            if x.shape[0] in (1, 3, 4):     # C H W
                x = x[None]
            else:                            # H W C
                x = x.permute(2, 0, 1)[None]
        elif x.ndim == 4:
            pass
        else:
            raise ValueError(f"Unsupported shape: {tuple(x.shape)}")
        return x

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    ref = to_bchw(ref).float().to(device)
    src = to_bchw(src).float().to(device)

    if ref.shape[0] != 1 or src.shape[0] != 1:
        raise ValueError("Batch size must be 1 for this helper.")

    _, C, H, W = ref.shape

    # If channel mismatch, fall back to grayscale for both
    if src.shape[1] != C:
        ref = ref.mean(dim=1, keepdim=True)
        src = src.mean(dim=1, keepdim=True)
        C = 1

    if mask_ref is None:
        mask_ref = torch.ones((1, 1, H, W), device=device)
    else:
        mask_ref = to_bchw(mask_ref).float().to(device)
        mask_ref = (mask_ref > 0).float()

    # Normalize for stable geometry optimization (optional but usually helps)
    ref_n = (ref - ref.mean(dim=(-2, -1), keepdim=True)) / (ref.std(dim=(-2, -1), keepdim=True) + 1e-6)
    src_n = (src - src.mean(dim=(-2, -1), keepdim=True)) / (src.std(dim=(-2, -1), keepdim=True) + 1e-6)

    ones_src = torch.ones((1, 1, src.shape[2], src.shape[3]), device=device)

    # ---- bounded parameterization for near-similarity + small shear ----
    rot_max = rotation_max_deg * torch.pi / 180.0

    u_theta = torch.zeros((), device=device, requires_grad=True)
    u_tx    = torch.zeros((), device=device, requires_grad=True)
    u_ty    = torch.zeros((), device=device, requires_grad=True)
    u_sx    = torch.zeros((), device=device, requires_grad=True)
    u_sy    = torch.zeros((), device=device, requires_grad=True)
    u_sh    = torch.zeros((), device=device, requires_grad=True)

    params = [u_theta, u_tx, u_ty, u_sx, u_sy, u_sh]

    def build_theta_2x3():
        theta = torch.tanh(u_theta) * rot_max
        tmax = 0.3 # = 2 * 0.05

        tx = torch.tanh(u_tx) * tmax
        ty = torch.tanh(u_ty) * tmax

        sx = scale_min + (scale_max - scale_min) * torch.sigmoid(u_sx)
        sy = scale_min + (scale_max - scale_min) * torch.sigmoid(u_sy)
        sh = torch.tanh(u_sh) * shear_max

        c, s = torch.cos(theta), torch.sin(theta)

        # 2x2 pieces
        R = torch.stack([torch.stack([c, -s]),
                         torch.stack([s,  c])])
        S = torch.diag(torch.stack([sx, sy]))
        Sh = torch.stack([torch.stack([torch.tensor(1.0, device=device), sh]),
                          torch.stack([torch.tensor(0.0, device=device), torch.tensor(1.0, device=device)])])

        A = R @ S @ Sh

        M = torch.zeros((2, 3), device=device)
        M[:2, :2] = A
        M[0, 2] = tx
        M[1, 2] = ty
        return M

    def closed_form_light(ref_img, src_warp, w):
        """
        Weighted least squares per-channel for y ≈ a*x + b.
        ref_img, src_warp: (1,C,H,W), w: (1,1,H,W)
        Returns a,b shaped (1,C,1,1)
        """
        w = w.clamp(0, 1)
        wsum = w.sum(dim=(-2, -1), keepdim=True).clamp_min(eps)  # (1,1,1,1)

        # Expand weights over channels
        wc = w.expand(-1, ref_img.shape[1], -1, -1)  # (1,C,H,W)

        mx = (wc * src_warp).sum(dim=(-2, -1), keepdim=True) / wsum
        my = (wc * ref_img).sum(dim=(-2, -1), keepdim=True) / wsum

        x0 = src_warp - mx
        y0 = ref_img - my

        varx = (wc * x0 * x0).sum(dim=(-2, -1), keepdim=True) / wsum
        cov  = (wc * x0 * y0).sum(dim=(-2, -1), keepdim=True) / wsum

        a = cov / (varx + eps)
        b = my - a * mx
        return a, b

    optimizer = torch.optim.LBFGS(params, lr=lr, max_iter=iters, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad(set_to_none=True)

        theta = build_theta_2x3()[None]              # (1,2,3)
        grid = F.affine_grid(theta, (1, C, H, W), align_corners=False)

        src_w = F.grid_sample(src_n, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        valid = F.grid_sample(ones_src, grid, mode="nearest", padding_mode="zeros", align_corners=False)

        w = (valid > 0.5).float() * mask_ref         # (1,1,H,W)

        # <-- strongest possible light fit for current warp
        a, b = closed_form_light(ref_n, src_w, w)
        src_adj = a * src_w + b

        diff = ref_n - src_adj
        loss_map = torch.sqrt(diff * diff + charb_eps * charb_eps)
        loss = (loss_map * w).sum() / (w.sum() + eps)

        loss.backward()
        return loss

    optimizer.step(closure)

    # ---- final outputs (apply light on warped ORIGINAL src, not normalized) ----
    with torch.no_grad():
        theta = build_theta_2x3()[None]
        grid = F.affine_grid(theta, (1, C, H, W), align_corners=False)

        src_w = F.grid_sample(src, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        valid = F.grid_sample(ones_src, grid, mode="nearest", padding_mode="zeros", align_corners=False)
        w = (valid > 0.5).float() * mask_ref

        # Light fit in ORIGINAL intensity space (stronger & more meaningful)
        a, b = closed_form_light(ref, src_w, w)
        src_adj = a * src_w + b

        out = src_adj[0].permute(1, 2, 0).cpu() if C > 1 else src_adj[0, 0].cpu()
        mask = (w[0, 0] > 0.5).to(torch.uint8).cpu()

        params_out = {
            "theta_2x3": build_theta_2x3().detach().cpu(),
            "a": a[0, :, 0, 0].detach().cpu(),
            "b": b[0, :, 0, 0].detach().cpu(),
        }

    return out, mask, params_out


def photometric_fit_affine(I_ref, I_src, mask=None, eps=1e-12):
    """
    Fit global photometric transform: I_src' = a * I_src + b
    minimizing squared error over mask (or all pixels).

    Works for grayscale or RGB (fits per-channel if RGB).

    Returns:
      I_src_corr : corrected I_src (float32)
      a, b       : scalars (grayscale) or shape (C,) for RGB
    """
    ref = np.asarray(I_ref, dtype=np.float32)
    src = np.asarray(I_src, dtype=np.float32)

    if ref.shape[:2] != src.shape[:2]:
        raise ValueError(f"Spatial shapes must match: {ref.shape} vs {src.shape}")

    if mask is None:
        m = np.ones(ref.shape[:2], dtype=bool)
    else:
        m = (np.asarray(mask) > 0)

    if ref.ndim == 2:
        # grayscale
        x = src[m].reshape(-1)
        y = ref[m].reshape(-1)

        # Solve min ||a x + b - y||^2
        mx, my = x.mean(), y.mean()
        vx = np.mean((x - mx) ** 2) + eps
        cov = np.mean((x - mx) * (y - my))
        a = cov / vx
        b = my - a * mx

        src_corr = a * src + b
        return src_corr.astype(np.float32), float(a), float(b)

    elif ref.ndim == 3:
        C = ref.shape[2]
        a = np.zeros(C, dtype=np.float32)
        b = np.zeros(C, dtype=np.float32)
        src_corr = np.empty_like(src, dtype=np.float32)

        for c in range(C):
            x = src[..., c][m].reshape(-1)
            y = ref[..., c][m].reshape(-1)
            mx, my = x.mean(), y.mean()
            vx = np.mean((x - mx) ** 2) + eps
            cov = np.mean((x - mx) * (y - my))
            a[c] = cov / vx
            b[c] = my - a[c] * mx
            src_corr[..., c] = a[c] * src[..., c] + b[c]

        return src_corr.astype(np.float32), a, b

    else:
        raise ValueError("I_ref/I_src must be 2D (grayscale) or 3D (H,W,C).")


def _to_ecc_gray(img):
    """Convert input to grayscale float32 suitable for cv2.findTransformECC."""
    x = np.asarray(img)

    # If RGB/RGBA -> grayscale
    if x.ndim == 3:
        if x.shape[2] == 4:
            x = cv.cvtColor(x, cv.COLOR_BGRA2GRAY) if x.dtype == np.uint8 else x[..., :3]
        if x.ndim == 3:  # still color
            # Handle both uint8 and float inputs
            if x.dtype == np.uint8:
                x = cv.cvtColor(x, cv.COLOR_BGR2GRAY)
            else:
                # assume RGB/BGR doesn't matter much; do a luminance-style mix
                x = 0.2989 * x[..., 0] + 0.5870 * x[..., 1] + 0.1140 * x[..., 2]

    # Now single channel
    x = x.astype(np.float32)

    # Optional: normalize helps ECC stability (not required, but often improves convergence)
    x = x - x.mean()
    s = x.std()
    if s > 1e-6:
        x = x / s

    return x

def align_restricted_affine(
    I_ref, I_src,
    max_deg=15.0,
    max_shear=0.05,   # ~ small skew
    allow_aniso_scale=False,
    nfev=200
):
    ref = _to_ecc_gray(I_ref.astype(np.float32))
    src = _to_ecc_gray(I_src.astype(np.float32))


    # normalize (helps)
    ref = (ref - ref.mean()) / (ref.std() + 1e-6)
    src = (src - src.mean()) / (src.std() + 1e-6)

    H, W = ref.shape
    yy, xx = np.mgrid[0:H, 0:W]

    # params: [theta, tx, ty, sx, sy, shear]
    # If not allow_aniso_scale, we tie sx=sy.
    def warp(p):
        theta, tx, ty, sx, sy, sh = p
        c, s = np.cos(theta), np.sin(theta)

        if not allow_aniso_scale:
            sy = sx

        # Rotation * scale
        R = np.array([[c, -s],
                      [s,  c]], dtype=np.float32)
        S = np.array([[sx, 0.0],
                      [0.0, sy]], dtype=np.float32)

        # Simple shear in x by y (you can choose other forms)
        Sh = np.array([[1.0, sh],
                       [0.0, 1.0]], dtype=np.float32)

        A = R @ S @ Sh  # 2x2
        return A, tx, ty

    def residuals(p):
        A, tx, ty = warp(p)
        xp = A[0,0]*xx + A[0,1]*yy + tx
        yp = A[1,0]*xx + A[1,1]*yy + ty
        Iw = map_coordinates(src, [yp, xp], order=1, mode='nearest')
        return (ref - Iw).ravel()

    max_rad = np.deg2rad(max_deg)

    # initial guess: no rot/scale/shear, no translation
    p0 = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 0.0], dtype=np.float64)

    # bounds
    lo = np.array([-max_rad, -W, -H, 0.5, 0.5, -max_shear], dtype=np.float64)
    hi = np.array([ max_rad,  W,  H, 2.0, 2.0,  max_shear], dtype=np.float64)

    res = least_squares(residuals, p0, bounds=(lo, hi), method="trf", max_nfev=nfev)

    # build final 2x3 matrix
    A, tx, ty = warp(res.x)
    M = np.array([[A[0,0], A[0,1], tx],
                  [A[1,0], A[1,1], ty]], dtype=np.float32)

    I_aligned = cv.warpAffine(I_src.astype(np.float32), M, (W, H), flags=cv.INTER_LINEAR)
    return I_aligned, M, res

def align_affine_ecc_with_init(I_ref, I_src, n_iters=200, eps=1e-6, gauss=3):
    ref = _to_ecc_gray(I_ref.astype(np.float32))
    src = _to_ecc_gray(I_src.astype(np.float32))

    # ensure grayscale
    if ref.ndim == 3: ref = cv.cvtColor(ref, cv.COLOR_BGR2GRAY)
    if src.ndim == 3: src = cv.cvtColor(src, cv.COLOR_BGR2GRAY)

    # normalize for stability
    refn = (ref - ref.mean()) / (ref.std() + 1e-6)
    srcn = (src - src.mean()) / (src.std() + 1e-6)

    if gauss and gauss > 0:
        refn = cv.GaussianBlur(refn, (0,0), gauss)
        srcn = cv.GaussianBlur(srcn, (0,0), gauss)

    # phase correlation gives (dx, dy) s.t. src shifted by (dx,dy) aligns to ref
    (dx, dy), _ = cv.phaseCorrelate(refn, srcn)

    warp = np.eye(2, 3, dtype=np.float32)
    warp[0, 2] = dx
    warp[1, 2] = dy

    criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, n_iters, eps)

    try:
        cc, warp = cv.findTransformECC(
            templateImage=refn,
            inputImage=srcn,
            warpMatrix=warp,
            motionType=cv.MOTION_EUCLIDEAN,
            criteria=criteria
        )
    except cv.error:
        # fallback: try translation-only (often succeeds when affine fails)
        warp = warp.copy()
        cc, warp = cv.findTransformECC(
            templateImage=refn,
            inputImage=srcn,
            warpMatrix=warp,
            motionType=cv.MOTION_TRANSLATION,
            criteria=criteria
        )

    aligned = cv.warpAffine(
        I_src.astype(np.float32),
        warp,
        (I_ref.shape[1], I_ref.shape[0]),
        flags=cv.INTER_LINEAR + cv.WARP_INVERSE_MAP
    )
    return aligned, warp

def _process_tile_worker(args):
    """
    Pickleable multiprocessing worker.
    Loads image/frag/mask via memmap for low RAM + no pickling big arrays.
    """
    (y0, y1, x0, x1,
     image_path,
     frag_items,   # list of (key, frag_path, mask_path)
     debug,
     out_dir) = args

    image = np.asarray(cv.imread(image_path, cv.IMREAD_UNCHANGED))
    tile = image[y0:y1, x0:x1]

    best_mse = None
    best_key = None
    best_frag_tile = None
    b_f = None
    # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # pi_metric = iqa.create_metric('lpips', device=device)

    for key, frag_path, mask_path in frag_items:
        frag = np.load(frag_path)
        mask = np.load(mask_path)

        m = mask[:, :, 0] if mask.ndim == 3 else mask
        if not np.all(m[y0:y1, x0:x1]):
            continue

        frag_tile = frag[y0:y1, x0:x1]
        #frag_tile_o = frag[y0:y1, x0:x1]

        #reg_tile, A, _ = align_restricted_affine(tile, frag_tile, max_deg=15, max_shear=0.06)
        opt_tile, mask , _ = align_affine_with_closedform_light_lbfgs(tile, frag_tile)
        opt_tile = np.asarray(opt_tile)
        mask = np.asarray(mask)[..., None].repeat(3, axis=2)
        # opt_tile, _, _ = photometric_fit_affine(tile, frag_tile)
        # reg_tile, A = align_affine_ecc_with_init(tile, opt_tile, n_iters=200, eps=1e-6, gauss=3)

        mse = np.square((tile - opt_tile) * mask).mean()
        # t_tile = torch.from_numpy(tile / 255).permute(2, 0, 1).float().unsqueeze(0).to(device)
        # t_frag = torch.from_numpy(frag_tile / 255).permute(2, 0, 1).float().unsqueeze(0).to(device)
        # score = pi_metric(t_tile, t_frag)
        #mse = score
        print(mse)

        if best_mse is None or mse < best_mse:
            best_mse = mse
            best_key = key
            if debug:
                best_frag_tile = opt_tile
                b_f = opt_tile

    if debug and best_mse is not None and best_mse > 0.0 and best_frag_tile is not None:
        diff = ((np.abs(tile - best_frag_tile) ** 0.5)  / ( 255 ** 0.5)) * 255
        debug_concat = np.concatenate((tile, best_frag_tile, diff, frag_tile), axis=0)
        out = debug_concat
        if out.dtype != np.uint8:
            out = np.clip(out, 0, 255).astype(np.uint8)

        os.makedirs(out_dir, exist_ok=True)
        cv.imwrite(os.path.join(out_dir, f"tile_{y0}_{x0}_{best_key}_{best_mse}.jpg"), out)

    return (y0, x0, best_mse, best_key)


def scale_homography(H_orig, w, h, W, H):
    """
    Scale a homography matrix from an original reference size (w,h)
    to a new size (W,H).

    Args:
        H_orig (np.ndarray): 3x3 original homography
        w (float): original width
        h (float): original height
        W (float): new width
        H (float): new height

    Returns:
        np.ndarray: 3x3 scaled homography
    """
    # compute scale factors
    sx = W / w
    sy = H / h

    # scale matrices
    S_toNew = np.array([
        [sx,  0,  0],
        [ 0, sy,  0],
        [ 0,  0,  1]
    ])

    S_toOld = np.array([
        [1/sx,    0,   0],
        [   0, 1/sy,   0],
        [   0,    0,   1]
    ])

    # apply scaling
    H_scaled = S_toNew @ H_orig @ S_toOld

    # normalize so bottom-right = 1 for stability
    if H_scaled[2,2] != 0:
        H_scaled /= H_scaled[2,2]

    return H_scaled

class Tester:
    def __init__(self, config):
        self.debug = True
        self.config = config
        self.roi = {'minH': 7100, 'maxH': 8100, 'minW': 0, 'maxW': 500}
        self.brisque = iqa.create_metric("brisque", device='cuda')


    def run(self):


        os.makedirs('./plots/tiles', exist_ok=True)
        #final_img =  np.load(f"./metrics/{self.config.exp_name}/final_img.npy")
        final_img_path = f'metrics/polokoule/final_stitch.png'
        with open(f"metrics/polokoule/homog.pkl", "rb") as f:
            self.homog = pickle.load(f)
        self.frag_paths = os.listdir(f"metrics/polokoule/images")
        #iqa_metrics()

        #results = self.run_parallel_tiles_futures(final_img_path, self.lo_frag_paths, 100, 100, 6)
        results = self.run_single_process_tiles(final_img_path, self.frag_paths, 200, 200)

    def tile_fully_in_fragment(self, mask, y0, y1, x0, x1):
        """
        True if ALL pixels of the tile are inside the fragment mask.
        """
        if mask.ndim == 3:
            mask = mask[:, :, 0]

        tile = mask[y0:y1, x0:x1]
        return np.all(tile)

    def iqa_metrics(self, args, homog):

        (y0, y1, x0, x1,
         image_path,
         frag_paths,  # list of (key, frag_path, mask_path)
         debug,
         out_dir) = args

        image = np.asarray(cv.imread(image_path, cv.IMREAD_UNCHANGED))
        tile = image[y0:y1, x0:x1]
        tile_torch = torch.from_numpy(tile).unsqueeze(0).float()
        self.brisque()


        # del image
        #
        # for idx, frag_path in enumerate(frag_paths):
        #     frag_homog = homog[idx]
        #     warped_fragment, frag_mask = self.stitcher.warp_image(frag_homog, frag_path)
        #
        #     m = frag_mask[:, :, 0] if frag_mask.ndim == 3 else frag_mask
        #     if not np.all(m[y0:y1, x0:x1]):
        #         continue
        #
        #     frag_tile = warped_fragment[y0:y1, x0:x1]





    def run_single_process_tiles(self, image_path, lo_frag_paths, th, tw):
        image = np.asarray(cv.imread(image_path, cv.IMREAD_UNCHANGED))
        H, W = image.shape[:2]

        # Make pickleable list of fragment file paths (same format as parallel version)
        frag_items = []
        for key, val in lo_frag_paths.items():
            frag_items.append((key, f"{val[0]}.npy", f"{val[1]}.npy"))

        tiles = [
            (y0, min(y0 + th, H), x0, min(x0 + tw, W))
            for y0 in range(self.roi['minH'], self.roi['maxH'], th)
            for x0 in range(self.roi['minW'], self.roi['maxW'], tw)
        ]

        out_dir = "./plots/tiles"
        os.makedirs(out_dir, exist_ok=True)

        tasks = [
            (y0, y1, x0, x1, image_path, frag_items, self.debug, out_dir)
            for (y0, y1, x0, x1) in tiles
        ]

        results = []
        for t in tasks:
            #results.append(_process_tile_worker(t))
            self.iqa_metrics(t)

        results.sort(key=lambda r: (r[0], r[1]))
        for (y0, x0, best_mse, best_key) in results:
            if best_mse is None:
                print(f"Tile {y0},{x0}: no candidates")
            else:
                print(f"Tile {y0},{x0}: MSE {best_mse} | best {best_key}")

        return results


    def run_parallel_tiles_futures(self, image_path, lo_frag_paths, th, tw, max_workers=None):
        image = np.asarray(cv.imread(image_path))
        H, W = image.shape[:2]

        # Make pickleable list of fragment file paths
        frag_items = []
        for key, val in lo_frag_paths.items():
            frag_items.append((key, f"{val[0]}.npy", f"{val[1]}.npy"))

        tiles = [(y0, min(y0 + th, H), x0, min(x0 + tw, W))
                 for y0 in range(0, H, th)
                 for x0 in range(0, W, tw)]

        out_dir = "./plots/tiles"

        tasks = [
            (y0, y1, x0, x1, image_path, frag_items, self.debug, out_dir)
            for (y0, y1, x0, x1) in tiles
        ]

        results = []
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(_process_tile_worker, t) for t in tasks]
            for f in as_completed(futures):
                results.append(f.result())

        results.sort(key=lambda r: (r[0], r[1]))
        for (y0, x0, best_mse, best_key) in results:
            if best_mse is None:
                print(f"Tile {y0},{x0}: no candidates")
            else:
                print(f"Tile {y0},{x0}: MSE {best_mse} | best {best_key}")

        return results


    # def cif_image_tiles(self, image, tile_size, cand_bits=None):
    #     def tile_fully_in_fragment(mask, y0, y1, x0, x1):
    #         """
    #         True if ALL pixels of the tile are inside the fragment mask.
    #         """
    #         if mask.ndim == 3:
    #             mask = mask[:, :, 0]
    #
    #         tile = mask[y0:y1, x0:x1]
    #         return np.all(tile)
    #
    #     H, W = image.shape[:2]
    #     th, tw = tile_size
    #     oh, ow = 50, 50
    #     for y0 in range(0, H, th):
    #         y1 = min(y0 + th, H)
    #         for x0 in range(0, W, tw):
    #             x1 = min(x0 + tw, W)
    #             tile = image[y0:y1, x0:x1]
    #             mses = []
    #
    #             debug_tiles = []
    #             for key, val in self.lo_frag_paths.items():
    #                 frag = np.load(f"{val[0]}.npy")
    #                 mask = np.load(f"{val[1]}.npy")
    #
    #                 frag_tile_o = frag[y0:y1, x0:x1]
    #                 if not tile_fully_in_fragment(mask, y0, y1, x0, x1):
    #                     continue
    #
    #                 reg_tile, A = align_affine_ecc(frag_tile, tile)
    #
    #                 mse =  (np.square(tile - reg_tile)).mean(axis=None)
    #                 mses.append(mse)
    #                 debug_tiles = [tile, frag_tile]
    #
    #             if self.debug and np.min(mses) > 0.0:
    #                 tile = debug_tiles[0]
    #                 frag_tile = debug_tiles[1]
    #                 debug_concat = np.concatenate((tile, reg_tile, tile - reg_tile), axis=0)
    #                 cv.imwrite(f"./plots/tiles/tile_{y0}_{x0}_{key}.jpg", debug_concat)
    #             print(f"MSE: {np.min(mses)} | from {len(mses)}")


def compose_configs(args):
    """
    Composes and merges configuration files from different sources including
    input configuration, default configuration, and preset configurations.

    The function performs the following steps:
    1. Loads the input configuration file from the specified input path.
    2. Loads the default configuration file.
    3. Loads a preset configuration file based on the preset name in the input
       configuration, or falls back to a default preset if none is specified.
    4. Merges the default and preset configurations.
    5. Merges the resulting configuration with the input configuration.
    6. Updates the configuration with specific paths for output and input
       directories.
    7. Creates the output directory if it does not exist.

    By combining configurations in the above manner, the function produces a
    final configuration object that can be used for further processing.

    Parameters:
        args (Namespace): A Namespace object containing the following attributes:
            - input (str): The path to the input directory.
            - output (str): The path to the output directory.

    Raises:
        FileNotFoundError: Raised if the input configuration file does not exist
            at the specified input path.

    Returns:
        DictConfig: The final composed and merged configuration object.
    """
    logger = logging.getLogger('INITIALIZE')
    # Loading input config
    if not os.path.exists(osp.join(args.input, 'config.yaml')):
        logger.error(f"Config file {osp.join(args.input, 'config.yaml')} not found")
        raise FileNotFoundError(f"Config file not found: {osp.join(args.input, 'config.yaml')}")
    logger.info(f"Loading input config from {osp.join(args.input, 'config.yaml')}")
    input_cfg = OmegaConf.load(osp.join(args.input, 'config.yaml'))

    # Load default config
    logger.info(f"Loading default config from configs/presets/default.yaml")
    default_cfg = OmegaConf.load("configs/presets/default.yaml")
    OmegaConf.set_struct(default_cfg, False)

    # Load preset if not specified load p_normal
    if hasattr(input_cfg, "preset_name"):
        logger.info(f"Loading preset {input_cfg.preset_name}")
        preset_cfg = OmegaConf.load(f"configs/presets/{input_cfg.preset_name}.yaml")
    else:
        logger.warning(f"! Preset not specified ! Loading p_normal.yaml")
        preset_cfg = OmegaConf.load(f"configs/presets/p_normal.yaml")
    OmegaConf.set_struct(preset_cfg, False)

    # Merge preset wit default
    preset_merged = OmegaConf.merge(default_cfg, preset_cfg)

    # Merge input cfg
    logger.info("Merging input cfg")
    OmegaConf.set_struct(input_cfg, False)
    config = OmegaConf.merge(preset_merged, input_cfg)
    logger.info("Config:\n%s", pprint.pformat(config))
    # Create output dir and adjust paths
    logger.info(f"Output folder: {osp.join(args.output, config.exp_name)}")
    config['output_folder'] = osp.join(args.output, config.exp_name)
    os.makedirs(config['output_folder'], exist_ok=True)

    logger.info(f"Input folder: {args.input}")
    config['input_folder'] = args.input

    return config

def args_process():
    parser = argparse.ArgumentParser(description="Process two input paths.")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to the input directory or file")
    parser.add_argument("--output", "-o", type=str, required=True, help="Path to the output directory or file")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        parser.error(f"Input path does not exist: {args.input}")

    return args

def create_dirs(config):

    logger = logging.getLogger('INITIALIZE')
    logger.info("Creating cache dirs")
    os.makedirs("plots", exist_ok=True)
    os.makedirs("cache", exist_ok=True)
    os.makedirs("cache/homogs", exist_ok=True)
    os.makedirs("cache/flows", exist_ok=True)
    logger.info("Cache dirs created")


def main():
    """
    This function serves as the main entry point for the application, handling initial setup,
    configuration processing, logging setup, and launching the core application functionality.

    Arguments:
        None

    Raises:
        OSError: If an error occurs while accessing or creating directories.
        FileNotFoundError: If a required file for configuration is missing.

    Returns:
        None
    """
    # Parse args
    args = args_process()
    # Clear temp_init
    with open(osp.join(args.output, "init_log.txt"), 'w'):
        pass  # just open and close to truncate
    # Temporary logger for init
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(osp.join(args.output, "init_log.txt")),
            logging.StreamHandler()],
        force = True,
    )
    # Make output dir
    os.makedirs(args.output, exist_ok=True)
    # Merge configs, allows for specification of hand-picked parameters
    config = compose_configs(args)

    shutil.copy(osp.join(args.output, "init_log.txt"), osp.join(config['output_folder'], "output_log.txt"))
    os.remove(osp.join(args.output, "init_log.txt"))
    # Reconfigure configs
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(osp.join(config['output_folder'], "output_log.txt"), mode="a"),
            logging.StreamHandler()],
        force = True,
    )
    # Load configs
    # Crete caching dirs
    create_dirs(config)
    # Run main stitcher
    app = Tester(config)
    app.run()

if __name__ == "__main__":
    main()
