import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = '10000000000'
import os.path as osp
import argparse
import shutil
import math
import json
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
import ptlflow
from ptlflow.utils.io_adapter import IOAdapter
from ptlflow.utils import flow_utils
from scipy.ndimage import map_coordinates
from scipy.optimize import least_squares
from omegaconf import OmegaConf

from vismatch import get_matcher, available_models
from modules.LightEqual import *

import pyiqa as iqa


def np_flow_to_img(flows):

    flow_img = flow_utils.flow_to_rgb(flows)
    # OpenCV uses BGR format
    flow_bgr_npy = cv.cvtColor(flow_img, cv.COLOR_RGB2BGR)

    return flow_bgr_npy

def compute_flow_misalignment_score(img1, img2, model):
    """
    Computes a misalignment score between two images using optical flow.

    Score is LOW for pure global shifts (uniform flow).
    Score is HIGH for misalignments, local jumps, or structural inconsistencies.

    Returns a dict with the scalar score and diagnostic sub-metrics.
    """

    # Load images (BGR -> keep as-is, IOAdapter handles conversion)

    input1 = img1[0].permute(1, 2, 0).cpu().numpy()
    input2 = img2[0].permute(1, 2, 0).cpu().numpy()
    # Prepare inputs
    io_adapter = IOAdapter(model, input1.shape[:2])
    inputs = io_adapter.prepare_inputs([input1, input2])

    # Run inference
    import torch
    with torch.no_grad():
        predictions = model(inputs)

    # flows shape: (1, 1, 2, H, W) — two channels: dx (u) and dy (v)
    flows = predictions["flows"][0, 0]  # shape (2, H, W)
    u = flows[0].cpu().numpy()  # horizontal displacement
    v = flows[1].cpu().numpy()  # vertical displacement

    # --- Flow magnitude map ---
    magnitude = np.sqrt(u**2 + v**2)  # shape (H, W)

    # --- Sub-metrics ---

    # 1. Spatial variance of magnitude: near 0 for pure shifts, high for misalignments
    mag_variance = float(np.var(magnitude))

    # 2. Gradient of magnitude: penalizes sharp spatial jumps
    grad_x = np.gradient(magnitude, axis=1)
    grad_y = np.gradient(magnitude, axis=0)
    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    mean_gradient = float(np.mean(grad_magnitude))

    # 3. Outlier ratio: fraction of pixels with magnitude >> median (local jumps)
    median_mag = float(np.median(magnitude))
    threshold = median_mag + 3 * float(np.std(magnitude))
    outlier_ratio = float(np.mean(magnitude > threshold))

    # 4. Mean magnitude (informational — a global shift has high mean but low variance)
    mean_magnitude = float(np.mean(magnitude))

    # --- Composite score ---
    # Normalize sub-metrics and combine. Weights are tunable.
    # mag_variance and mean_gradient are in pixel units — normalize by mean_magnitude
    # to make the score scale-invariant.
    eps = 1e-6
    normalized_variance = mag_variance / (mean_magnitude**2 + eps)
    normalized_gradient = mean_gradient / (mean_magnitude + eps)

    # Composite: weighted sum (tune weights to your use case)
    score = (
        0.3 * normalized_variance +
        0.4 * normalized_gradient +
        0.3 * outlier_ratio
    ) * 100
    stitched_flow_img = np_flow_to_img(flows.permute(1, 2, 0).detach().cpu().numpy()) / 255
    # print(score)
    # cv.imshow("flow", np.hstack(( np.astype(input1 * 255, np.uint8), stitched_flow_img, np.astype(input2 * 255, np.uint8))))
    # cv.waitKey()
    return {
        "score": score,             # HIGH = bad alignment / jumps; LOW = pure shift
        "mean_magnitude": mean_magnitude,
        "magnitude_variance": mag_variance,
        "mean_gradient": mean_gradient,
        "outlier_ratio": outlier_ratio,
        'flow_img': stitched_flow_img,
    }


# --- Example usage -

def _process_tile_worker(args):
    """
    Pickleable multiprocessing worker.
    Loads image/frag/mask via memmap for low RAM + no pickling big arrays.
    """
    (y0, y1, x0, x1,
     image_path,
     frag_items,   # list of (key, frag_path, mask_path)
     debug,
     out_dir, metrics_calculator, model) = args

    image = np.asarray(cv.imread(image_path, cv.IMREAD_UNCHANGED))
    H, W = image.shape[:2]

    # padded region for alignment
    pad = 20
    py0 = max(0, y0 - pad)
    py1 = min(y1 + pad, H)
    px0 = max(0, x0 - pad)
    px1 = min(x1 + pad, W)

    stitch_tile_padded = image[py0:py1, px0:px1]
    stitch_tile = image[y0:y1, x0:x1]

    # coordinates of NON-PADDED tile inside padded tile
    inner_y0 = y0 - py0
    inner_y1 = inner_y0 + (y1 - y0)
    inner_x0 = x0 - px0
    inner_x1 = inner_x0 + (x1 - x0)

    metrics_results = {'cw_ssim': {'score': 0},
                       'lpips': {'score': 0},
                       'roma': {'score': 0},
                       'flow': {'score': 1000}}

    for key, frag, mask in frag_items:

        m = mask[:, :, 0] if mask.ndim == 3 else mask
        if not np.all(m[y0:y1, x0:x1]):
            continue

        frag_tile_padded = frag[py0:py1, px0:px1]
        frag_tile = frag[y0:y1, x0:x1]


        # ---- CROP BACK TO NON-PADDED TILE ----

        t_stitch_tile = torch.from_numpy(stitch_tile / 255).permute((2, 0, 1)).unsqueeze(0).float()
        t_frag_tile = torch.from_numpy(frag_tile / 255).permute((2, 0, 1)).unsqueeze(0).float()

        flow_res = compute_flow_misalignment_score(t_stitch_tile, t_frag_tile, model)
        flow_score = flow_res['score']
        flow_img = flow_res['flow_img']

        # CW_SSIM
        cw_ssim_score = metrics_calculator['cw_ssim'](t_stitch_tile, t_frag_tile)

        # Roma
        roma_score = metrics_calculator['roma'](t_frag_tile[0], t_stitch_tile[0])

        # LPIPS
        lpips_score = metrics_calculator['lpips'](t_stitch_tile, t_frag_tile)

        if metrics_results['cw_ssim']['score'] < cw_ssim_score :
            metrics_results['cw_ssim']['score'] = cw_ssim_score.item()
            metrics_results['cw_ssim']['best_tile'] = frag_tile
            metrics_results['cw_ssim']['best_tile_key'] = key

        if metrics_results['roma']['score'] < roma_score:
            metrics_results['roma']['score'] = roma_score
            metrics_results['roma']['best_tile'] = frag_tile
            metrics_results['roma']['best_tile_key'] = key

        if metrics_results['lpips']['score'] < lpips_score:
            metrics_results['lpips']['score'] = lpips_score.item()
            metrics_results['lpips']['best_tile'] = frag_tile
            metrics_results['lpips']['best_tile_key'] = key

        if metrics_results['flow']['score'] > flow_score:
            metrics_results['flow']['score'] = flow_score
            metrics_results['flow']['best_tile'] = frag_tile
            metrics_results['flow']['best_tile_key'] = key
            metrics_results['flow']['flow_img'] = flow_img


    metrics_results['offset'] = (y0, x0)
    metrics_results['stitch'] = stitch_tile

    return metrics_results

from PIL import Image, ImageDraw, ImageFont

def stack_images_with_metrics(images, metrics, output_path="output.png", upscale=2):
    """
    images: list of 3 numpy arrays (H, W, C) or (H, W)
    metrics: list of 3 strings
    """

    h, w = images[0].shape[:2]
    uw, uh = w * upscale, h * upscale

    panels = []
    for img, metric in zip(images, metrics):
        # Normalize to uint8
        if img.dtype != np.uint8:
            img = (np.clip(img, 0, 1) * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)

        # Grayscale to BGR
        if img.ndim == 2:
            img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)

        panel = cv.resize(img, (uw, uh), interpolation=cv.INTER_CUBIC)

        font = cv.FONT_HERSHEY_DUPLEX
        font_scale = max(0.5, uh / 600)
        thickness = max(1, int(font_scale * 2))
        margin = 10

        # Shadow
        cv.putText(panel, metric, (margin + 2, margin + 2 + int(font_scale * 30)),
                    font, font_scale, (0, 0, 0), thickness + 2, cv.LINE_AA)
        # Text
        cv.putText(panel, metric, (margin, margin + int(font_scale * 30)),
                    font, font_scale, (0, 255, 255), thickness, cv.LINE_AA)

        panels.append(panel)

    combined = np.hstack(panels)
    cv.imwrite(output_path, combined)
    print(f"Saved to {output_path}, size: {combined.shape[1]}x{combined.shape[0]}")
    return combined

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


def scale_homography_two(H_small,
                     dst_shape_small, dst_shape_full,
                     src_shape_small=None, src_shape_full=None):
    """
    Scales a homography matrix from low-res to full-res using OpenCV image shapes.

    Args:
        H_small         : 3x3 homography estimated at low resolution.
        dst_shape_small : Tuple (height, width) of the base image at LOW resolution.
        dst_shape_full  : Tuple (height, width) of the base image at FULL resolution.
        src_shape_small : Tuple (height, width) of the fragment image at LOW res. (Optional)
        src_shape_full  : Tuple (height, width) of the fragment image at FULL res. (Optional)

    Returns:
        np.ndarray: The scaled 3x3 homography matrix for full resolution.
    """
    # 1. Unpack shapes (OpenCV format is Height, Width)
    dst_small_h, dst_small_w = dst_shape_small[:2]
    dst_full_h, dst_full_w = dst_shape_full[:2]

    # 2. Calculate scale factors.
    # Remember: Width corresponds to the X-axis, Height corresponds to the Y-axis.
    dst_scale_x = dst_full_w / dst_small_w
    dst_scale_y = dst_full_h / dst_small_h

    S_dst = np.array([
        [dst_scale_x, 0, 0],
        [0, dst_scale_y, 0],
        [0, 0, 1]
    ], dtype=np.float64)

    # 3. Handle source (fragment) scaling if both shapes are provided
    if src_shape_small is not None and src_shape_full is not None:
        src_small_h, src_small_w = src_shape_small[:2]
        src_full_h, src_full_w = src_shape_full[:2]

        src_scale_x = src_full_w / src_small_w
        src_scale_y = src_full_h / src_small_h
    else:
        src_scale_x = 1.0
        src_scale_y = 1.0

    S_src_inv = np.array([
        [1.0 / src_scale_x, 0, 0],
        [0, 1.0 / src_scale_y, 0],
        [0, 0, 1]
    ], dtype=np.float64)

    # 4. Compute and normalize the full-resolution homography
    H_full = S_dst @ H_small @ S_src_inv
    H_full = H_full / H_full[2, 2]

    return H_full

class RomaMetric:
    def __init__(self):
        self.matcher = get_matcher('roma', device="cuda")

    def __call__(self, img_from, img_to):
        result = self.matcher(img_from, img_to)
        ratio = result["num_inliers"] / max(len(result["matched_kpts0"]), 1)
        return ratio

class Tester:
    def __init__(self, config):
        N = 2
        self.debug = True
        self.config = config
        self.config.final_res = (int(self.config.final_res[0] / N), int(self.config.final_res[1] / N))
        self.roi = {'minH': 21000, 'maxH': 23000, 'minW': 0, 'maxW': 1000}
        for k,v in self.roi.items():
            self.roi[k] = int(v/N)
        #self.roi = {'minH': int(11000/N), 'maxH': int(11200/N), 'minW': int(3300/N), 'maxW': int(4800/N)}
        self.rect_size = (2770, 3768)

        self.metrics_calculator = {'cw_ssim': iqa.create_metric('cw_ssim', device='cuda'),
                                   'roma': RomaMetric(),
                                   'lpips': iqa.create_metric('lpips', device='cuda'),}

    def warp_image(self, homography, frag_path, res=None):
        """
        Warp the given image fragment using a homography matrix.

        This function applies a homography transformation to warp an image fragment
        into a specific coordinate space. The resulting warped image and a corresponding
        mask are returned. The mask indicates valid regions within the warped image.
        If no resolution is provided, the function uses the default final resolution
        from configuration, otherwise it uses the specified resolution.

        Parameters:
            homography: numpy.ndarray
                A 3x3 homography matrix for warping the image.
            frag_path: str
                The path to the image fragment to be warped.
            res: tuple[int, int], optional
                The target resolution (height, width) for the warped image. Defaults to None.

        Returns:
            tuple[numpy.ndarray, numpy.ndarray]
                The warped image as a numpy array and its corresponding mask.
        """
        # Get the corner of the final image

        x_min, y_min = (0, 0)
        if res is None:
            x_max, y_max = (self.config.final_res[1], self.config.final_res[0])
        else:
            x_max, y_max = (res[1], res[0])
        # Compute translation homography to shift images to positive coordinates
        translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])

        # Load fragment
        fragment = cv.imread(frag_path).astype(np.float32)
        masking_array = np.ones_like(fragment, np.uint8)
        # Calculate the homography
        H = translation @ homography
        # Apply warping based on homography
        warped = cv.warpPerspective(fragment, H, (x_max - x_min, y_max - y_min))
        mask = cv.warpPerspective(masking_array, H, (x_max - x_min, y_max - y_min))
        # Mask
        mask = (mask > 0)

        return warped, mask

    def run(self):

        homogs_load = json.load(open(f"{self.config.input_folder}/homographies.json", "r"))
        os.makedirs('./plots/tiles', exist_ok=True)
        #final_img =  np.load(f"./metrics/{self.config.exp_name}/final_img.npy")
        final_img_path =  os.path.join(self.config.input_folder, "final_stitch.png")
        self.frag_paths = self.config.fragments

        self.frag_names = os.listdir(f"{self.config.fragments}/images")
        self.frag_names.sort()
        #iqa_metrics()

        self.warped_frags = {}
        for frag_n in self.frag_names:
            if frag_n == self.config.ref_name: continue
            if frag_n not in homogs_load.keys():
                print(f"Fragment {frag_n} not found in homogs_load. Skipping...")
                continue
            frag_h = np.asarray(homogs_load[frag_n])
            frag_h = scale_homography_two(frag_h, self.rect_size, (self.config.final_res[0], self.config.final_res[1]))
            frag_warped, mask = self.warp_image(frag_h, f"{self.config.fragments}/images/{frag_n}")
            self.warped_frags[frag_n] = (frag_warped, mask)
            # cv.namedWindow("w", cv.WINDOW_NORMAL)
            #
            # # 2. Set the window to your desired fixed size (width, height)
            # #    For example, 800 pixels wide and 600 pixels tall.
            # cv.resizeWindow("w", 800, 600)
            #
            # # 3. Show the image in the configured window
            # cv.imshow("w", frag_warped)
            # cv.waitKey(0)
        #results = self.run_parallel_tiles_futures(final_img_path, self.lo_frag_paths, 100, 100, 6)
        results = self.run_single_process_tiles(final_img_path, self.warped_frags, 400, 400)

    def run_single_process_tiles(self, image_path, warped_frags, th, tw):

        image = np.asarray(cv.imread(image_path, cv.IMREAD_UNCHANGED))
        image = cv.resize(image, (self.config.final_res[1], self.config.final_res[0]))
        H, W = image.shape[:2]

        res_ref_path = f"cache/res_ref.png"
        cv.imwrite(res_ref_path, image)

        # Make pickleable list of fragment file paths (same format as parallel version)
        frag_items = []
        for key, val in warped_frags.items():
            frag_items.append((key, val[0], val[1]))

        #self.roi = {'minH': 0, 'maxH': H, 'minW': 0, 'maxW': W}
        tiles = [
            (y0, min(y0 + th, H), x0, min(x0 + tw, W))
            for y0 in range(self.roi['minH'], self.roi['maxH'], th)
            for x0 in range(self.roi['minW'], self.roi['maxW'], tw)
        ]

        out_dir = "./plots/tiles"
        os.makedirs(out_dir, exist_ok=True)

        # Load model
        model = ptlflow.get_model("memflow_t", ckpt_path="sintel")
        model.eval()

        tasks = [
            (y0, y1, x0, x1, res_ref_path, frag_items, self.debug, out_dir, self.metrics_calculator, model)
            for (y0, y1, x0, x1) in tiles
        ]

        results = []
        for t in tasks:
            results.append(_process_tile_worker(t))

        self.save_results(results, out_path=out_dir, upscale=2)

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

        return results

    def save_results(self, results, out_path, best_threshold=None, worst_threshold=None, upscale=None):


        # Save worst in both
        save_path_worst = osp.join(out_path, 'worst')
        save_comp_path = osp.join(out_path, 'comparison')
        save_path_best = osp.join(out_path, 'best')
        os.makedirs(save_comp_path, exist_ok=True)
        os.makedirs(save_path_worst, exist_ok=True)
        os.makedirs(save_path_best, exist_ok=True)

        for idx, result in enumerate(results):

            if 'best_tile' not in result['flow'].keys():
                continue

            # Decide only based on flow score
            flow_score = result['flow']['score']



            if flow_score > 10.0:
                save_name = f"flow_{flow_score:.3f}_{result['offset'][0]}_{result['offset'][1]}.png"
                cv.imwrite(osp.join(save_path_worst, save_name),result['stitch'])
                stack_images_with_metrics([result['stitch'], result['flow']['best_tile'], result['flow']['flow_img']],
                                          [f'{flow_score:.3f}', f'{flow_score:.3f}', f'{flow_score:.3f}'],
                                          output_path=osp.join(save_comp_path,
                                                               f"worst_{result['offset'][0]}_{result['offset'][1]}.png"))


            if flow_score < 5.0:
                save_name = f"flow_{flow_score:.3f}_{result['offset'][0]}_{result['offset'][1]}.png"
                cv.imwrite(osp.join(save_path_best, save_name), result['stitch'])
                stack_images_with_metrics([result['stitch'], result['flow']['best_tile'], result['flow']['flow_img']],
                                          [f'{flow_score:.3f}', f'{flow_score:.3f}', f'{flow_score:.3f}'],
                                          output_path=osp.join(save_comp_path,
                                                               f"best_{result['offset'][0]}_{result['offset'][1]}.png"))


            # if result['cw_ssim']['score'] == 0:
            #     continue
            # roma_score = result['roma']['score']
            # ssim_score = max(0.1, math.log((result['cw_ssim']['score'] + 1e-5 )**0.8) + 1)
            # flow_score = result['flow']['score']
            # decision_score = round(roma_score * ssim_score,3)
            #
            # stack_images_with_metrics([result['stitch'], result['roma']['best_tile'], result['cw_ssim']['best_tile'], result['flow']['best_tile'], result['flow']['flow_img']],
            #                           [f'{decision_score}', f'{result['roma']['score']:.3f}',f'{result['cw_ssim']['score']:.3f}', f'{result['flow']['score']:.3f}', f'{result['flow']['score']:.3f}' ],
            #                           output_path=osp.join(save_comp_path, f"{result['offset'][0]}_{result['offset'][1]}.png"))
            #
            # if flow_score > 10.0:
            #     if upscale is not None:
            #         uh, uw = result['stitch'].shape[:2]
            #         uh, uw = int(uh * upscale), int(uw * upscale)
            #         save_name = f"flow_{flow_score:.3f}_{result['offset'][0]}_{result['offset'][1]}.png"
            #         save_name = f"cwrm_{result['cw_ssim']['score']:.3f}_{result['roma']['score']:.3f}_{flow_score:.3f}_{result['offset'][0]}_{result['offset'][1]}.png"
            #         cv.imwrite(osp.join(save_path, save_name),
            #                    cv.resize(result['stitch'], (uw, uh), interpolation=cv.INTER_CUBIC))

            # # CW SSIM
            # if decision_score < 0.9:
            #
            #     if upscale is not None:
            #         uh, uw = result['stitch'].shape[:2]
            #         uh, uw = int(uh * upscale), int(uw * upscale)
            #
            #         save_name = f"cwrm_{result['cw_ssim']['score']:.3f}_{result['roma']['score']:.3f}_{result['offset'][0]}_{result['offset'][1]}.png"
            #         cv.imwrite(osp.join(save_path, save_name),
            #                    cv.resize(result['stitch'], (uw, uh), interpolation=cv.INTER_CUBIC))
            #         uh, uw = result['roma']['best_tile'].shape[:2]
            #         uh, uw = int(uh * upscale), int(uw * upscale)
            #
            #         save_name = f"no_{result['cw_ssim']['score']:.3f}_{result['roma']['score']:.3f}_{result['offset'][0]}_{result['offset'][1]}.png"
            #         cv.imwrite(osp.join(save_path, save_name), cv.resize(result['roma']['best_tile'], (uw, uh), interpolation=cv.INTER_CUBIC))
            #     else:
            #         save_name = f"cwrm_{result['cw_ssim']['score']:.3f}_{result['roma']['score']:.3f}_{result['offset'][0]}_{result['offset'][1]}.png"
            #         cv.imwrite(osp.join(save_path, save_name), result['stitch'])
            #
            #         save_name = f"no_{result['cw_ssim']['score']:.3f}_{result['roma']['score']:.3f}_{result['offset'][0]}_{result['offset'][1]}.png"
            #         cv.imwrite(osp.join(save_path, save_name), result['roma']['best_tile'])
            #
            #     saved.append(idx)

        # # Save worst in ROMA
        # for idx, result in enumerate(results):
        #     if idx in saved: continue
        #
        #     if
        #     save_name = f"rm_{result['cw_ssim']['score']:.3f}_{result['roma']['score']:.3f}_{result['offset'][0]}_{result['offset'][1]}.png"
        #     cv.imwrite(osp.join(save_path, save_name),  result['stitch'])
        #     saved.append(idx)

        # saved = []
        # save_path = osp.join(out_path, 'best_both')
        # os.makedirs(save_path, exist_ok=True)
        # for idx, result in enumerate(results):
        #
        #
        #     roma_score = result['roma']['score']
        #     ssim_score = max(0.1, math.log(result['cw_ssim']['score'] ** 0.8) + 1)
        #     decision_score = round(roma_score * ssim_score, 3)
        #     # CW SSIM
        #     if decision_score > 0.98:
        #         if upscale is not None:
        #             uh, uw = result['cw_ssim']['best_tile'].shape[:2]
        #             uh, uw = int(uh * upscale), int(uw * upscale)
        #
        #             save_name = f"cwrm_{result['cw_ssim']['score']}_{result['roma']['score']}_{result['offset'][0]}_{result['offset'][1]}.png"
        #             cv.imwrite(osp.join(save_path, save_name), cv.resize(result['cw_ssim']['best_tile'], (uw, uh), interpolation=cv.INTER_CUBIC))
        #         else:
        #             save_name = f"cwrm_{result['cw_ssim']['score']}_{result['roma']['score']}_{result['offset'][0]}_{result['offset'][1]}.png"
        #             cv.imwrite(osp.join(save_path, save_name), result['cw_ssim']['best_tile'])
        #         saved.append(idx)


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
    config['fragments'] = args.frag

    return config

def args_process():
    parser = argparse.ArgumentParser(description="Process two input paths.")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to the input directory or file")
    parser.add_argument("--output", "-o", type=str, required=True, help="Path to the output directory or file")
    parser.add_argument("--frag", "-f", type=str, required=True, help="Path to the fragments directory")

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
