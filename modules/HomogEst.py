import cv2
from kornia.feature import LoFTR
from lightglue.utils import load_image, rbd
import cv2 as cv
import os
import sys
from lightglue import viz2d
from typing_extensions import override


from utils.homography_optimalizer import *
from lightglue import LightGlue, SuperPoint
import torch

from torchvision.transforms import v2
import logging

from vismatch import get_matcher, available_models
from vismatch.viz import plot_matches

def load_imgs(img_paths):
    # Load images in super point and light glue format
    imgs = {}
    for p in img_paths:
        _, filename = os.path.split(p)
        filename, _ = filename.split(".")
        img = load_image(p)
        imgs[int(filename)] = img
    return imgs

def load_img(path):
    img = load_image(path)
    return img

def tensor_to_numpy_image(tensor):
    """
    Convert a (1, 1, H, W) tensor to a uint8 grayscale numpy image.
    """
    img = tensor.squeeze().cpu().numpy()
    img = (img * 255).astype(np.uint8)
    return img

def save_loftr_matches(
    img0, img1, kpts0, kpts1, filename="loftr_matches.jpg", colors=None):
    """
    Saves matching keypoints between two images with connecting lines and visualizes them side by side.

    The function takes two input images and their respective sets of matching keypoints, overlays lines connecting
    corresponding points, and combines the visuals into one stacked image. It saves the result as an image file.

    Parameters:
        img0: numpy.ndarray
            First input image, either grayscale or color.
        img1: numpy.ndarray
            Second input image, either grayscale or color.
        kpts0: list of tuple
            List of keypoints (x, y) from the first image.
        kpts1: list of tuple
            List of keypoints (x, y) from the second image.
        filename: str, optional
            File name for the saved visualized matches image. Default is "loftr_matches.jpg".
        colors: list of tuple or None, optional
            List of RGB color tuples for each match. If None, all matches will be drawn in green.

    Raises:
        None

    Returns:
        None
    """
    if len(img0.shape) == 2:
        img0 = cv2.cvtColor(img0, cv2.COLOR_GRAY2BGR)
    if len(img1.shape) == 2:
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)

    # Stack images side by side
    h = max(img0.shape[0], img1.shape[0])
    w0, w1 = img0.shape[1], img1.shape[1]
    canvas = np.zeros((h, w0 + w1, 3), dtype=np.uint8)
    canvas[: img0.shape[0], :w0] = img0
    canvas[: img1.shape[0], w0:] = img1

    # Draw lines between matches
    for i, (pt0, pt1) in enumerate(zip(kpts0, kpts1)):
        x0, y0 = map(int, pt0)
        x1, y1 = map(int, pt1)
        color = (0, 255, 0) if colors is None else colors[i]
        cv2.line(canvas, (x0, y0), (x1 + w0, y1), color, 1, cv2.LINE_AA)
        cv2.circle(canvas, (x0, y0), 2, color, -1)
        cv2.circle(canvas, (x1 + w0, y1), 2, color, -1)

    cv2.imwrite(filename, canvas)


def _save_debug_imgs(dat1, dat2, matches, path="./plots/matches.jpg"):
    axes = viz2d.plot_images([dat1[0], dat2[0]], adaptive=False, dpi=500)
    viz2d.plot_matches(dat1[1], dat2[1], color="lime", lw=0.2)
    #viz2d.add_text(0, f'Stop after {matches["stop"]} layers | {len(dat1[2])},  {len(dat2[2])}', fs=20)
    viz2d.save_plot(path)


def _save_key_imgs(dat1, dat2, path="./plots/matches.jpg"):
    axes = viz2d.plot_images([dat1[0], dat2[0]], adaptive=False, dpi=500)
    viz2d.plot_keypoints([dat1[2], dat2[2]])
    viz2d.add_text(0, f' kpts = {len(dat1[2])},  {len(dat2[2])}', fs=20)
    viz2d.save_plot(path)




def _resize(img: torch.Tensor, new_h: int, new_w: int) -> torch.Tensor:
    return torch.nn.functional.interpolate(
        img.unsqueeze(0),
        size=(new_h, new_w),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    ).squeeze(0)

class HomogEstimator:
    """
    The HomogEstimator class is used for estimating homographies between images. It utilizes deep feature
    matchers and keypoint extractors to compute matching between fragments and a reference image. This class
    is particularly useful for tasks involving image alignment, stitching, or feature-based homography.

    HomogEstimator encompasses functionalities like feature extraction using SuperPoint, matching using
    LightGlue, and advanced handling of homography estimation, including optimization and debugging support.
    It supports retry mechanisms, scale adjustments, and advanced debugging to augment feature-matching
    process robustness.
    """
    def __init__(self, config):
        self.logger = logging.getLogger("HOMOG")
        self.config = config
        self.images = []
        # Init feature extractor
        torch.set_grad_enabled(False)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.kornia = False
        self.ransac_conf: float = 0.95
        self.ransac_reproj_thresh: float = 3

        if hasattr(config, "eval_res"):
            self.resize = config.eval_res
        else:
            self.resize = None

        self.scale_x = 1.0
        self.scale_y = 1.0

        if self.config.homog.feature_matcher == 'LightGlue':
            self.extractor = SuperPoint(max_num_keypoints=config.homog.max_feat_points).eval().to(self.device)
            # Init feature matcher
            self.matcher = LightGlue(features="superpoint",depth_confidence=-1,
            width_confidence=-1).eval().to(self.device)

        elif self.config.homog.feature_matcher == 'superpoint-lightglue':
            self.kornia = True
            self.extractor = None
            self.matcher = get_matcher('superpoint-lightglue', device="cuda")

        elif self.config.homog.feature_matcher == 'dedode-lightglue':
            self.kornia = True
            self.extractor = None
            self.matcher = get_matcher('dedode-lightglue', device="cuda")

        elif self.config.homog.feature_matcher == 'rdd-lightglue':
            self.kornia = True
            self.extractor = None
            self.matcher = get_matcher('rdd-lightglue', device='cuda')

        elif self.config.homog.feature_matcher == 'omniglue':
            self.kornia = True
            self.extractor = None
            self.matcher = get_matcher('omniglue', device="cpu")

        elif self.config.homog.feature_matcher == 'loftr':
            self.kornia = True
            self.extractor = None
            self.matcher = get_matcher('eloftr', device="cuda")

        elif self.config.homog.feature_matcher == 'xfeat':
            self.kornia = True
            self.extractor = None
            self.matcher = get_matcher('xfeat', device="cuda")

        elif self.config.homog.feature_matcher == 'roma':
            self.kornia = True
            self.extractor = None
            self.matcher = get_matcher('roma', device="cuda")

        elif self.config.homog.feature_matcher in available_models:
            self.extractor = None
            self.matcher = get_matcher(self.config.homog.feature_matcher, device="cuda")
        else:
            raise NotImplementedError
        # Init optimizer
        if config.homog.do_optimization:
            print("Loading optimizer :", config.homog.optimizer)
            if config.homog.optimizer == 'marek':
                self.homog_opt = HomographyOptimizer(max_matches=config.homog.max_matches)
            else:
                raise NotImplementedError

        self.debug = config.homog.debug

    def adjust_fragment(self, fragment, scale):
        """
        Adjusts the detail present in fragment image by resizing it to a given scale and then returning
        it back to original dimensions.

        Parameters:
        fragment : torch.Tensor
            Input tensor with shape (channels, height, width). The fragment to be adjusted.
        scale : float
            Scale factor by which the fragment's dimensions are resized.

        Returns:
        torch.Tensor
            The adjusted fragment tensor, scaled and restored to the original dimensions.
        """
        h, w = fragment.shape[1:]

        fragment = fragment.unsqueeze(0)
        frag_low = torch.nn.functional.interpolate(fragment, scale_factor=scale, mode='bilinear', align_corners=False, antialias=True)
        fragment = torch.nn.functional.interpolate(frag_low, size=(h,w), mode='bilinear', align_corners=False, antialias=True)
        return fragment.squeeze(0)


    def match_details(self, frag_img, adjust_scale=0):
        """
        Matches details from the provided fragment image with an optional scale adjustment.

        This function applies a scaling adjustment to the fragment image if the configuration
        contains a 'relative_scale' parameter. It computes a scaling factor and adjusts the image
        accordingly. If the parameter is missing or its value is invalid, appropriate logging is performed.

        Parameters:
            frag_img: The fragment image to be adjusted.
            adjust_scale: Optional scale adjustment to modify the relative scaling factor (default is 0).

        Returns:
            The adjusted fragment image after applying the computed scaling, or the original
            fragment image if scaling could not be applied.

        Raises:
            Does not explicitly raise an error, but logs any issues to the provided logger instance.
        """
        if hasattr(self.config, "relative_scale"):
            try:
                scale =  1 / np.clip(self.config.relative_scale + adjust_scale + 1e-5, 1, 100)
                frag_img = self.adjust_fragment(frag_img, scale)
            except:
                self.logger.error("Invalid value in relative scale")

        else:
            self.logger.error("Missing 'relative_scale' parameter unable to autofix feature matching")

        return frag_img

    def retry_matching(self, feats_ref, frag_path, ref_img=None):
        """
        Retries computing the matching between a reference feature set and a fragment
        image multiple times with different scale adjustments. If enough matches are
        found, computes the homography matrix and matched keypoints.

        Parameters:
        feats_ref : tensor
            The extracted feature descriptors for the reference image.
        frag_path : str
            Path to the fragment image whose features are to be matched with the
            reference.

        Returns:
        tuple
            A tuple containing:
            - H: Homography matrix computed between the reference image and the
              fragment image, or None if sufficient matches are not found.
            - mkpts: Matched keypoints between the reference and fragment images,
              or None if sufficient matches are not found.
        """

        # Retry with different relative scales
        H, mkpts = None, None
        scale_modifiers = [1, -1, 2, -2, 4, -4]
        for scale_modifier in scale_modifiers:
            if self.extractor is not None:
                frag_img = load_image(frag_path)
                frag_img = self.match_details(frag_img, adjust_scale=scale_modifier)
                feats_frag = self.extractor.extract(frag_img.to(self.device))
                matches_a_b = self.matcher({"image0": feats_ref, "image1": feats_frag})
            if matches_a_b['matches'][0].shape[0] > self.config.homog.min_matches:
                H, _, mkpts = self.get_homography(feats_ref, feats_frag, matches_a_b, (0, -1))
                break

        return H, mkpts

    def retry_match_vismatch(self, frag_path, ref_img):
        #TODO SIMPLIFY THIS
        res = None
        scale_modifiers = [1, -1, 2, -2, 4, -4]
        for scale_modifier in scale_modifiers:
            frag_img = self.matcher.load_image(frag_path)
            frag_img = self.match_details(frag_img, scale_modifier)
            self.images = [np.asarray(ref_img.permute(1, 2, 0).cpu()), np.asarray(frag_img.permute(1, 2, 0).cpu())]

            if hasattr(self.config, 'eval_res'):
                ref_img, frag_img, scale_x, scale_y = self.resize_pair_to_max_res(ref_img, frag_img,
                                                                                  max_res=self.config.eval_res)
                results = self.matcher(frag_img, ref_img)
                kpts_ref = self.restore_kpts_to_original(results["inlier_kpts1"], scale_x, scale_y)
                kpts_frag = self.restore_kpts_to_original(results["inlier_kpts0"], scale_x, scale_y)
            else:
                results = self.matcher(frag_img, ref_img)
                kpts_ref = results["inlier_kpts1"]
                kpts_frag = results["inlier_kpts0"]

            results["inlier_kpts1"] = kpts_ref
            results["inlier_kpts0"] = kpts_frag
            res = results
            if results["inlier_kpts0"].shape[0] > self.config.homog.min_matches:

                break
        return res

    def resize_cubic(self, img: torch.Tensor, size):
        """
        Resize a torch image using bicubic interpolation.

        Args:
            img: Tensor of shape (C,H,W) or (B,C,H,W)
            size: (new_h, new_w)

        Returns:
            Resized tensor (same dimensionality as input)
        """
        is_batched = img.dim() == 4

        if not is_batched:
            img = img.unsqueeze(0)  # add batch dim

        resized = torch.nn.functional.interpolate(
            img,
            size=size,
            mode="bicubic",
            align_corners=False
        )

        if not is_batched:
            resized = resized.squeeze(0)

        return resized

    def match_fragments(self, feats_ref, frag_img, idx):
        """
        Attempts to match image fragments by comparing extracted features and computing
        homography.

        Args:
            feats_ref: Extracted features from the reference image.
            frag_img: The fragment of the image to be matched.
            idx: An identifier or index for tracking/logging purposes.

        Returns:
            A tuple containing:
                - Homography matrix (H) computed between the reference and fragment.
                - Keypoints associated with matches that were successfully matched.

        Raises:
            Any raised exceptions are not explicitly documented.
        """

        feats_frag = self.extractor.extract(frag_img.to(self.device))

        # Match features with reference
        matches_a_b = self.matcher({"image0": feats_ref, "image1": feats_frag})

        self.logger.info(
            f"[{idx}] Num. Features: {feats_frag['keypoints'].shape[1]} | Matches {matches_a_b['matches'][0].shape[0]}")

        # Compute homography
        H, m, mkpts = self.get_homography(feats_ref, feats_frag, matches_a_b, (0, idx))

        return H, mkpts

    def compute_scale_and_resize_ref(
            self,
            img_ref: torch.Tensor,
            max_res: int = 1024,
    ):
        _, H, W = img_ref.shape
        scale = min(max_res / H, max_res / W)

        if scale >= 1.0:
            return img_ref, 1.0, 1.0

        new_H = int(round(H * scale))
        new_W = int(round(W * scale))

        img_ref_resized = _resize(img_ref, new_H, new_W)
        scale_x = new_W / W
        scale_y = new_H / H

        return img_ref_resized, scale_x, scale_y

    def resize_fragment(self,
            img_frag: torch.Tensor,
            scale_x: float,
            scale_y: float,
    ):
        if scale_x == 1.0 and scale_y == 1.0:
            return img_frag

        _, H, W = img_frag.shape
        new_H = int(round(H * scale_y))
        new_W = int(round(W * scale_x))

        return _resize(img_frag, new_H, new_W)

    def restore_kpts_to_original(self,
            kpts,
            scale_x,
            scale_y,
    ):
        """
        Rescales keypoints from the resized image space back to the original
        image pixel space.

        Args:
            kpts:    (N, 2) array of (x, y) keypoints in resized space
            scale_x: Horizontal scale factor returned by resize_pair_to_max_res
            scale_y: Vertical scale factor returned by resize_pair_to_max_res

        Returns:
            kpts in original pixel space
        """
        return kpts / np.array([scale_x, scale_y])

    def register(self, ref_path: str, frag_paths: list[str]):
        """
        Registers fragments against the reference image by extracting and matching features,
        computing homographies, and retrying matching for failed attempts.

        Parameters:
        ref_path: str
            Path to the reference image.
        frag_paths: list[str]
            List of paths to fragment images to be registered.

        Returns:
        tuple[list, list, list]
            The method returns three values:
            - A list of computed homographies for each fragment with successful matching.
            - A list of corresponding matched keypoints.
            - A list of indices for fragments that failed the registration process.
        """
        # Get ref img
        self.logger.info(f"Loading reference from {ref_path}")



        # Extract features
        if self.extractor is not None:
            ref_img = load_image(ref_path)
            #ref_img = self.resize_cubic(ref_img, (resize, resize))
            feats_ref = self.extractor.extract(ref_img.to(self.device))
            self.logger.info(f"REF Num. Features: {feats_ref['keypoints'].shape[1]}")
        else:
            ref_img = self.matcher.load_image(ref_path)

        if self.resize is not None:
            ref_img, self.scale_x, self.scale_y = self.compute_scale_and_resize_ref(ref_img, max_res=self.resize)

        # Iterate over fragments and estimate homography
        homographies, corrs, to_del = [], [], []
        for idx, frag_path in enumerate(frag_paths):
            # Extract frag features
            self.logger.info(f"[{idx}] Loading fragment from {frag_path}")

            if self.extractor is not None:
                frag_img = load_img(frag_path)
                # Try to match details according to specified values in donfig
                frag_img = self.match_details(frag_img)
                self.images = [np.asarray(ref_img.permute(1, 2, 0).cpu()), np.asarray(frag_img.permute(1, 2, 0).cpu())]
                if self.resize is not None:
                    frag_img = self.resize_fragment(frag_img, self.scale_x, self.scale_y)

                # Match features with between ref and frag and compute homography
                H, mkpts = self.match_fragments(feats_ref, frag_img, idx)
                # If we were unable to compute homography or not enough point were match try to fix it
                if H is None or mkpts[0].shape[0] < self.config.homog.min_matches:
                    self.logger.warning(f"Autofix | Matcher was unable to estimate homography trying simple autofix")
                    H, mkpts = self.retry_matching(feats_ref, frag_path)
            else:
                frag_img = self.matcher.load_image(frag_path)
                frag_img = self.match_details(frag_img)
                self.images = [np.asarray(ref_img.permute(1, 2, 0).cpu()), np.asarray(frag_img.permute(1, 2, 0).cpu())]

                if self.resize is not None:
                    frag_img = self.resize_fragment(frag_img, self.scale_x, self.scale_y)

                    results = self.matcher(frag_img, ref_img)
                    kpts_ref =  self.restore_kpts_to_original(results["inlier_kpts1"], self.scale_x, self.scale_y)
                    kpts_frag = self.restore_kpts_to_original(results["inlier_kpts0"], self.scale_x, self.scale_y)
                else:
                    results = self.matcher(frag_img, ref_img)
                    kpts_ref =  results["inlier_kpts1"]
                    kpts_frag = results["inlier_kpts0"]

                mkpts = [kpts_frag, kpts_ref]
                H, mask = cv.findHomography(
                    kpts_frag,
                    kpts_ref,
                    cv.USAC_MAGSAC,
                    ransacReprojThreshold=self.ransac_reproj_thresh,
                    confidence=self.ransac_conf
                )


                if H is None or mkpts[0].shape[0] < self.config.homog.min_matches:
                    self.logger.warning(f"Autofix | Matcher was unable to estimate homography trying simple autofix")
                    results = self.retry_match_vismatch(frag_path, ref_img)
                    kpts_ref = results["inlier_kpts1"]
                    kpts_frag = results["inlier_kpts0"]
                    mkpts = [kpts_frag, kpts_ref]
                    H, mask = cv.findHomography(
                        kpts_frag,
                        kpts_ref,
                        cv.USAC_MAGSAC,
                        ransacReprojThreshold=self.ransac_reproj_thresh,
                        confidence=self.ransac_conf
                    )

                self.logger.info(
                    f"[{idx}] Num. Features: {results['all_kpts0'].shape[0]} | Matches {results['inlier_kpts0'].shape[0]}")
                if self.config.homog.debug:
                    plot_matches(frag_img, ref_img, results, save_path=f"./plots/matches_{idx}.jpeg")

            # If we fail to estimate homography after simple autofix performed than dont use that fragment
            if H is None:
                if hasattr(self.config, 'eval_res'):
                    self.upscale_homography(H, self.orig_ref_size ,(self.resize,self.resize))
                to_del.append(idx)
                continue
            homographies.append(H)
            corrs.append(mkpts)
            # Memory clean
            torch.cuda.empty_cache()
        return homographies, corrs, to_del

    def get_homography(self, feats1, feats2, matches12, pair):
        """
        Computes the homography matrix between two sets of features based on their matches.

        This method processes input feature dictionaries and corresponding matches to compute
        the homography matrix using RANSAC. The matched keypoints are extracted and saved as
        debug images if the debug mode is enabled in the configuration. If there are not
        enough points to compute the homography, it returns None values. Otherwise, the computed
        homography matrix, mask, and transformed keypoints are returned.

        Parameters:
            feats1 (dict): Dictionary of features for the first image. Should include "keypoints".
            feats2 (dict): Dictionary of features for the second image. Should include "keypoints".
            matches12 (dict): Dictionary containing the matches between keypoints of
                the first and second image. Should include "matches".
            pair (tuple[int, int]): Tuple of indices identifying the image pair.

        Returns:
            H (np.ndarray or None): The computed homography matrix, or None if computation failed.
            mask (np.ndarray or None): Mask of inliers and outliers, or None if computation failed.
            transformed_keypoints (tuple[np.ndarray, np.ndarray] or None):
                A tuple of arrays representing transformed keypoints from images 1 and 2,
                or None if computation failed.

        Raises:
            No exceptions are explicitly raised by this method, but it assumes valid input
            conforming to the specified types and structures for correct functionality.
        """
        # Reshape the input
        feats1, feats2, matches12 = [
            rbd(x) for x in [feats1, feats2, matches12]
        ]  # remove batch dimension
        # Extract Matched key points from key points lists for homography
        kpts1, kpts2, matches = feats1["keypoints"], feats2["keypoints"], matches12["matches"]
        m_kpts1, m_kpts2 = kpts1[matches[..., 0]], kpts2[matches[..., 1]]

        # Save image with matched features
        if self.config.homog.debug:
            _save_debug_imgs([self.images[0], m_kpts1, kpts1],
                             [self.images[1], m_kpts2, kpts2],
                             matches12,
                             path=f"./plots/matches_{pair[0]}_{pair[1]}.jpeg")
            _save_key_imgs([self.images[0], m_kpts1, kpts1],
                            [self.images[1], m_kpts2, kpts2],
                            path=f"./plots/kpts_{pair[0]}_{pair[1]}.jpeg")

        if len(np.asarray(m_kpts2.cpu())) < 20 or len(np.asarray(m_kpts1.cpu())) < 20:
            print("Not enough points")
            return None, None, None

        H, mask = cv.findHomography(np.asarray(m_kpts2.cpu()), np.asarray(m_kpts1.cpu()), cv.USAC_MAGSAC, ransacReprojThreshold=self.ransac_reproj_thresh,
                                    confidence=self.ransac_conf)
        return H, mask, (np.asarray(m_kpts2.cpu()), np.asarray(m_kpts1.cpu()))

    def upscale_homography(self, H_r, orig_size, resized_size):
        """
        Convert homography estimated on resized image
        to work on original resolution.

        Args:
            H_r: 3x3 homography estimated on resized image
            orig_size: (H, W) original
            resized_size: (H_r, W_r)

        Returns:
            H_full: homography for original resolution
        """
        H, W = orig_size
        Hr, Wr = resized_size

        sx = Wr / W
        sy = Hr / H

        S = np.array([
            [sx, 0, 0],
            [0, sy, 0],
            [0, 0, 1]
        ])

        S_inv = np.linalg.inv(S)

        H_full = S_inv @ H_r @ S
        return H_full
