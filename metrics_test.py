import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = '10000000000'
import os.path as osp
import argparse
import shutil
import math

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

from vismatch import get_matcher, available_models
from modules.LightEqual import *

import pyiqa as iqa

homogs_load = {
    "_MG_0362.JPG": [
        [
            0.31335198922589563,
            0.006341511152524633,
            2499.608366806162
        ],
        [
            0.005378849383839863,
            0.3192019788277783,
            2080.9750436446734
        ],
        [
            2.9708793378801474e-06,
            4.2379992531133153e-07,
            1.0
        ]
    ],
    "_MG_0361.JPG": [
        [
            0.3224542549497558,
            0.006823279705767507,
            3197.560959956914
        ],
        [
            0.01259602629984654,
            0.3268100133022859,
            2037.5822666101237
        ],
        [
            3.726652050174987e-06,
            1.4726164336281983e-06,
            1.0
        ]
    ],
    "_MG_0363.JPG": [
        [
            0.3236189596035262,
            0.022717793122490882,
            1837.4281493863386
        ],
        [
            0.005370817355333138,
            0.34206134191812043,
            2008.2802480296657
        ],
        [
            3.348311096062642e-06,
            5.770116062253942e-06,
            1.0
        ]
    ],
    "_MG_0370.JPG": [
        [
            0.30955056503113687,
            0.04642888301054227,
            2923.8234617271733
        ],
        [
            0.011436577990297889,
            0.3219000122947731,
            1553.210862312777
        ],
        [
            6.467461111990328e-06,
            1.106852178980209e-05,
            1.0
        ]
    ],
    "_MG_0369.JPG": [
        [
            0.30115007725760884,
            0.03438438512778632,
            2072.0634582418957
        ],
        [
            0.004566410236292263,
            0.3246684680803045,
            1445.4971578408254
        ],
        [
            3.132226931191358e-06,
            1.0392923691834576e-05,
            1.0
        ]
    ],
    "_MG_0360.JPG": [
        [
            0.304789696072951,
            0.027033687192250837,
            3637.867992035221
        ],
        [
            0.00177845460314975,
            0.32795546661204256,
            2085.3221597649713
        ],
        [
            1.173942707309822e-06,
            4.489857808965992e-06,
            1.0
        ]
    ],
    "_MG_0371.JPG": [
        [
            0.2453974477511394,
            0.04736302014501589,
            3749.2539771813804
        ],
        [
            -0.017672118009741954,
            0.3050311365278768,
            1575.9606371920545
        ],
        [
            -6.124852902256309e-06,
            9.03911330212638e-06,
            0.9999999999999999
        ]
    ],
    "_MG_0374.JPG": [
        [
            0.2713784189167475,
            0.010831428530083375,
            2156.886425886127
        ],
        [
            0.015400136634536873,
            0.2871092828053739,
            1054.3419471835405
        ],
        [
            1.9852962256263474e-06,
            6.7930717451741564e-06,
            1.0
        ]
    ],
    "_MG_0376.JPG": [
        [
            0.25986205770080295,
            -0.0012455237599964149,
            347.71524905005566
        ],
        [
            0.01208679846219046,
            0.2805999258237064,
            1143.380495392664
        ],
        [
            6.878417429677565e-07,
            6.511483955188236e-06,
            1.0
        ]
    ],
    "_MG_0375.JPG": [
        [
            0.25132118441310913,
            0.025775668002881048,
            1226.0978833474296
        ],
        [
            -0.012220973024580709,
            0.28067263392182557,
            1116.7589733781588
        ],
        [
            -3.264896537885547e-06,
            7.970573443105314e-06,
            0.9999999999999999
        ]
    ],
    "_MG_0379.JPG": [
        [
            0.26693616091070255,
            0.007315856858412047,
            875.3469674076629
        ],
        [
            -0.0037475896884522192,
            0.28783603078497444,
            443.7889540367356
        ],
        [
            -4.470892270878492e-06,
            3.0790281257610565e-06,
            1.0
        ]
    ],
    "_MG_0378.JPG": [
        [
            0.27372079674218974,
            0.002691789627787631,
            -75.67571362486952
        ],
        [
            -0.0006684919655069625,
            0.28980668826155975,
            378.87192237495333
        ],
        [
            -2.1123060164018823e-06,
            4.0136024404689e-06,
            0.9999999999999999
        ]
    ],
    "_MG_0366.JPG": [
        [
            0.30208062082799125,
            0.0092576504811732,
            -136.57342354889093
        ],
        [
            0.022832726551933212,
            0.3319158957743719,
            1362.6574009603
        ],
        [
            9.100941124713164e-06,
            1.2490919121667504e-05,
            1.0
        ]
    ],
    "_MG_0367.JPG": [
        [
            0.300728054197003,
            0.018460266733822193,
            647.4912083056756
        ],
        [
            0.01381887155576054,
            0.33044028646351065,
            1366.440144526241
        ],
        [
            4.978734802833722e-06,
            1.2607801224058762e-05,
            1.0
        ]
    ],
    "_MG_0380.JPG": [
        [
            0.2574486054790282,
            0.010146621532729168,
            1632.6997722765109
        ],
        [
            -0.011177784692585646,
            0.28047154497715315,
            457.0385297740884
        ],
        [
            -5.316998602058588e-06,
            7.390825406120036e-07,
            1.0
        ]
    ],
    "_MG_0365.JPG": [
        [
            0.33408906946908584,
            0.0072549241321726455,
            -29.477980215868723
        ],
        [
            0.009478951669194375,
            0.3344220438022248,
            1949.8602356212014
        ],
        [
            4.710831008373566e-06,
            -3.264812170834002e-06,
            1.0
        ]
    ],
    "_MG_0364.JPG": [
        [
            0.3060386267861837,
            0.0021496817892832374,
            1123.4297484015162
        ],
        [
            -0.0031923995167194596,
            0.31359135955156975,
            2025.9744001099505
        ],
        [
            -7.101436551037986e-07,
            -1.7059027258074908e-06,
            1.0
        ]
    ],
    "_MG_0368.JPG": [
        [
            0.2549135289857803,
            -0.09695658008549471,
            1431.0431288041166
        ],
        [
            -0.0009986305359193663,
            0.15065643315204344,
            1482.5443101540568
        ],
        [
            -1.2546336224229835e-06,
            -5.032421211865662e-05,
            1.0
        ]
    ],
    "_MG_0381.JPG": [
        [
            0.26000015919854247,
            0.007692179255641692,
            2434.455741716991
        ],
        [
            -0.008956417597586582,
            0.2785353295417881,
            485.47999881144256
        ],
        [
            -3.0471249132981976e-06,
            -1.4438578369008824e-06,
            1.0
        ]
    ],
    "_MG_0382.JPG": [
        [
            0.263816176778489,
            -0.000758481838321132,
            3264.2962369913507
        ],
        [
            -0.0003384573406254476,
            0.2851308070896135,
            399.42221080284173
        ],
        [
            -2.6599529982638697e-06,
            -1.472239173246098e-06,
            1.0
        ]
    ],
    "_MG_0372.JPG": [
        [
            0.18092728346640435,
            0.029497092774401622,
            3935.880133542559
        ],
        [
            -0.008842899377076776,
            0.27395187534780746,
            844.4864896200851
        ],
        [
            -1.565581403106911e-05,
            6.900653627705095e-06,
            1.0
        ]
    ],
    "_MG_0383.JPG": [
        [
            0.25332196538953594,
            -0.01701346222132456,
            3789.0446530291956
        ],
        [
            -0.0030864799687421386,
            0.2814176356102219,
            367.63633218528594
        ],
        [
            -4.628954146204741e-06,
            -4.938428498893218e-06,
            1.0
        ]
    ],
    "_MG_0373.JPG": [
        [
            0.28240224252895335,
            0.042449494415102484,
            2894.0085277753606
        ],
        [
            -0.0027570812209732998,
            0.2867439486053365,
            1049.4700664768336
        ],
        [
            5.290957250754908e-06,
            6.984833705268043e-06,
            1.0
        ]
    ],
    "_MG_0385.JPG": [
        [
            0.24373959770836037,
            0.014076554503266534,
            3245.6927779625266
        ],
        [
            -0.001021376498508491,
            0.27719695848634834,
            -73.86364323637999
        ],
        [
            -5.26727266214588e-06,
            2.3391119935838505e-06,
            1.0
        ]
    ],
    "_MG_0384.JPG": [
        [
            0.25956346565060195,
            -0.01615246396562202,
            3816.793756673182
        ],
        [
            0.005971918896896105,
            0.2713656454332083,
            -84.55416930563457
        ],
        [
            -2.8831880246903076e-07,
            -3.57386755476431e-06,
            1.0
        ]
    ],
    "_MG_0387.JPG": [
        [
            0.2590895385491501,
            0.007242421387648465,
            1417.5387816315738
        ],
        [
            0.0032574165224395616,
            0.278416332305639,
            -47.08683444729208
        ],
        [
            -4.011491007263634e-06,
            3.919624516481356e-06,
            1.0
        ]
    ],
    "_MG_0388.JPG": [
        [
            0.25241815247822685,
            0.004458914190344325,
            629.0066875063452
        ],
        [
            -0.006868975844853938,
            0.2664898797795504,
            5.582828123347634
        ],
        [
            -4.505122118054488e-06,
            -9.721622421567128e-07,
            1.0
        ]
    ],
    "_MG_0386.JPG": [
        [
            0.24827273710708614,
            -0.00364433005960266,
            2421.2125091976072
        ],
        [
            0.002503186898251658,
            0.2745307536161873,
            -70.71753868926787
        ],
        [
            -5.285561625935572e-06,
            -1.1847986668990601e-06,
            1.0
        ]
    ],
    "_MG_0377.JPG": [
        [
            0.262418186157483,
            -0.0059221310600452515,
            -150.2316242124095
        ],
        [
            0.014831989935225824,
            0.2813882977704562,
            1280.3020656162903
        ],
        [
            1.8638367424917428e-06,
            5.1228225732204386e-06,
            1.0
        ]
    ],
    "_MG_0389.JPG": [
        [
            0.2667515835442977,
            0.005547627899013016,
            -155.10616069420738
        ],
        [
            0.00045770237605120855,
            0.2795469620120058,
            -47.12087111467719
        ],
        [
            -3.2445571757459275e-07,
            7.447020488241755e-06,
            1.0
        ]
    ]
}

def _process_tile_worker(args):
    """
    Pickleable multiprocessing worker.
    Loads image/frag/mask via memmap for low RAM + no pickling big arrays.
    """
    (y0, y1, x0, x1,
     image_path,
     frag_items,   # list of (key, frag_path, mask_path)
     debug,
     out_dir, metrics_calculator) = args

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
                       'roma': {'score': 0}}

    for key, frag, mask in frag_items:

        m = mask[:, :, 0] if mask.ndim == 3 else mask
        if not np.all(m[y0:y1, x0:x1]):
            continue

        frag_tile_padded = frag[py0:py1, px0:px1]
        frag_tile = frag[y0:y1, x0:x1]


        # ---- CROP BACK TO NON-PADDED TILE ----

        t_stitch_tile = torch.from_numpy(stitch_tile / 255).permute((2, 0, 1)).unsqueeze(0).float()
        t_frag_tile = torch.from_numpy(frag_tile / 255).permute((2, 0, 1)).unsqueeze(0).float()

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
        self.roi = {'minH': int(11000/N), 'maxH': int(11200/N), 'minW': int(3300/N), 'maxW': int(4800/N)}
        self.rect_size = (3152, 5168)

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


        os.makedirs('./plots/tiles', exist_ok=True)
        #final_img =  np.load(f"./metrics/{self.config.exp_name}/final_img.npy")
        final_img_path = f'/home/dejvax/storage/brno12/scratch/Exposea/Exposea_p_200mpx_part3/2026_04_30:04_18_24/output/protifeudalne/final_stitch.png'

        self.frag_names = os.listdir(f"{self.config.input_folder}/images")
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
            frag_warped, mask = self.warp_image(frag_h, f"{self.config.input_folder}/images/{frag_n}")
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
        results = self.run_single_process_tiles(final_img_path, self.warped_frags, 100, 100)

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

        tasks = [
            (y0, y1, x0, x1, res_ref_path, frag_items, self.debug, out_dir, self.metrics_calculator)
            for (y0, y1, x0, x1) in tiles
        ]

        results = []
        for t in tasks:
            results.append(_process_tile_worker(t))

        self.save_results(results, out_path=out_dir)

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

    def save_results(self, results, out_path, best_threshold=None, worst_threshold=None):

        if best_threshold is None:
            best_threshold = {'cw_ssim': 0.9, 'lpips': 0.9, 'roma': 0.98}
        if worst_threshold is None:
            worst_threshold = {'cw_ssim': 0.8, 'lpips': 0.9, 'roma': 0.97}



        # Save worst in both
        save_path = osp.join(out_path, 'worst_both')
        save_comp_path = osp.join(out_path, 'comparison')
        os.makedirs(save_comp_path, exist_ok=True)
        os.makedirs(save_path, exist_ok=True)
        saved = []

        for idx, result in enumerate(results):

            roma_score = result['roma']['score']
            ssim_score = max(0.1, math.log(result['cw_ssim']['score']**0.8) + 1)
            decision_score = round(roma_score * ssim_score,3)

            stack_images_with_metrics([result['stitch'], result['roma']['best_tile'], result['cw_ssim']['best_tile']],
                                      [f'{decision_score}', f'{result['roma']['score']:.3f}',f'{result['cw_ssim']['score']:.3f}' ],
                                      output_path=osp.join(save_comp_path, f"{result['offset'][0]}_{result['offset'][1]}.png"))
            # CW SSIM
            if decision_score < 0.9:
                save_name = f"cwrm_{result['cw_ssim']['score']:.3f}_{result['roma']['score']:.3f}_{result['offset'][0]}_{result['offset'][1]}.png"
                cv.imwrite(osp.join(save_path, save_name), result['stitch'])

                save_name = f"no_{result['cw_ssim']['score']:.3f}_{result['roma']['score']:.3f}_{result['offset'][0]}_{result['offset'][1]}.png"
                cv.imwrite(osp.join(save_path, save_name), result['roma']['best_tile'])
                saved.append(idx)

        # Save worst in ROMA
        for idx, result in enumerate(results):
            if idx in saved: continue

            save_name = f"rm_{result['cw_ssim']['score']:.3f}_{result['roma']['score']:.3f}_{result['offset'][0]}_{result['offset'][1]}.png"
            cv.imwrite(osp.join(save_path, save_name),  result['stitch'])
            saved.append(idx)

        saved = []
        save_path = osp.join(out_path, 'best_both')
        os.makedirs(save_path, exist_ok=True)
        for idx, result in enumerate(results):

            roma_score = result['roma']['score']
            ssim_score = max(0.1, math.log(result['cw_ssim']['score'] ** 0.8) + 1)
            decision_score = round(roma_score * ssim_score, 3)
            # CW SSIM
            if decision_score > 0.98:
                save_name = f"cwrm_{result['cw_ssim']['score']}_{result['roma']['score']}_{result['offset'][0]}_{result['offset'][1]}.png"
                cv.imwrite(osp.join(save_path, save_name), result['cw_ssim']['best_tile'])
                saved.append(idx)


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
