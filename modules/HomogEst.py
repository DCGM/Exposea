import cv2
from lightglue.utils import load_image, rbd
import cv2 as cv
import os
from lightglue import viz2d
from utils.homography_optimalizer import *
from lightglue import LightGlue, SuperPoint
import torch

from torchvision.transforms import v2
import logging
import kornia.feature as KF

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
    Visualize and save LoFTR matches as a side-by-side image.

    Args:
        img0: First image (grayscale or BGR).
        img1: Second image (grayscale or BGR).
        kpts0: Nx2 array of keypoints from image0.
        kpts1: Nx2 array of keypoints from image1.
        filename: Where to save the image.
        colors: Optional list of BGR colors per match.
        show: If True, display using OpenCV.
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
    viz2d.add_text(0, f'Stop after {matches["stop"]} layers | {len(dat1[2])},  {len(dat2[2])}', fs=20)
    viz2d.save_plot(path)


def _save_key_imgs(dat1, dat2, path="./plots/matches.jpg"):
    axes = viz2d.plot_images([dat1[0], dat2[0]], adaptive=False, dpi=500)
    viz2d.plot_keypoints([dat1[2], dat2[2]])
    viz2d.add_text(0, f' kpts = {len(dat1[2])},  {len(dat2[2])}', fs=20)
    viz2d.save_plot(path)



class HomogEstimator:

    def __init__(self, config):
        self.logger = logging.getLogger("HOMOG")
        self.config = config
        # Init feature extractor
        torch.set_grad_enabled(False)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.extractor = SuperPoint(max_num_keypoints=config.homog.max_feat_points).eval().to(self.device)
        # Init feature matcher
        self.matcher = LightGlue(features="superpoint",depth_confidence=-1,
        width_confidence=-1).eval().to(self.device)

        # Init optimizer
        if config.homog.do_optimization:
            print("Loading optimizer :", config.homog.optimizer)
            if config.homog.optimizer == 'marek':
                self.homog_opt = HomographyOptimizer(max_matches=config.homog.max_matches)
            else:
                raise NotImplementedError

        self.debug = config.homog.debug

    def adjust_fragment(self, fragment, scale):
        h, w = fragment.shape[1:]

        fragment = fragment.unsqueeze(0)
        frag_low = torch.nn.functional.interpolate(fragment, scale_factor=scale, mode='bilinear', align_corners=False, antialias=True)
        fragment = torch.nn.functional.interpolate(frag_low, size=(h,w), mode='bilinear', align_corners=False, antialias=True)
        return fragment.squeeze(0)


    def register(self, ref_path: str, frag_paths: list[str]):
        # Get ref img
        self.logger.info(f"Loading reference from {ref_path}")
        ref_img = load_image(ref_path)
        # Extract features
        feats_ref = self.extractor.extract(ref_img.to(self.device))
        self.logger.info(f"REF Num. Features: {feats_ref['keypoints'].shape[1]}")
        # Iterate over fragments and estimate homography
        homographies = []
        corrs = []
        to_del = []
        for idx, frag_path in enumerate(frag_paths):
            # Extract frag features
            self.logger.info(f"[{idx}] Loading fragment from {frag_path}")
            frag_img = load_img(frag_path)

            if hasattr(self.config, "relative_scale"):
                try:
                    frag_img = self.adjust_fragment(frag_img, 1 / self.config.relative_scale)
                except:
                    self.logger.error("Invalid value in relative scale")

            elif hasattr(self.config, "frag_ref_dpi"):
                try:
                    frag_img = self.adjust_fragment(frag_img, 1 / (self.config.frag_ref_dpi[0] / self.config.frag_ref_dpi[1]))
                except:
                    self.logger.error("Invalid value in frag_ref_dpi")

            # For debug output
            if self.config.homog.debug:
                self.images = (ref_img, frag_img)

            feats_frag = self.extractor.extract(frag_img.to(self.device))
            # Find matches between images
            matches_a_b = self.matcher({"image0": feats_ref, "image1": feats_frag})
            self.logger.info(f"[{idx}] Num. Features: {feats_frag['keypoints'].shape[1]} | Matches {matches_a_b['matches'][0].shape[0]}")
            # Ger homography from image b to a
            H, m, mkpts = self.get_homography(feats_ref, feats_frag, matches_a_b, (0, idx))

            if H is None:
                to_del.append(idx)
                continue

            homographies.append(H)
            corrs.append(mkpts)

        return homographies, corrs, to_del

    def get_homography(self, feats1, feats2, matches12, pair):
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

        H, mask = cv.findHomography(np.asarray(m_kpts2.cpu()), np.asarray(m_kpts1.cpu()), cv.RANSAC, 5.0)
        return H, mask, (np.asarray(m_kpts2.cpu()), np.asarray(m_kpts1.cpu()))
