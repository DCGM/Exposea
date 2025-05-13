from lightglue.utils import load_image, rbd
import cv2 as cv
import os
from lightglue import viz2d
from marek.homography_optimalizer import *
from lightglue import LightGlue, SuperPoint
import torch
from memory_profiler import profile

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
        self.config = config
        # Init feature extractor
        torch.set_grad_enabled(False)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.extractor = SuperPoint(max_num_keypoints=config.homog.max_feat_points).eval().to(self.device)
        # Init feature matcher
        self.matcher = LightGlue(features="superpoint").eval().to(self.device)

        # Init optimizer
        if config.homog.do_optimization:
            print("Loading optimizer :", config.homog.optimizer)
            if config.homog.optimizer == 'marek':
                self.homog_opt = HomographyOptimizer(max_matches=config.homog.max_matches)
            else:
                raise NotImplementedError

        self.debug = config.homog.debug


    def register(self, ref_path: str, frag_paths: list[str]):
        # TODO Switch to this
        # Get ref img
        ref_img = load_image(ref_path)
        # Extract features
        feats_ref = self.extractor.extract(ref_img.to(self.device))

        # Iterate over fragments and estimate homography
        homographies = []
        corrs = []
        for idx, frag_path in enumerate(frag_paths):
            # Extract frag features
            frag_img = load_image(frag_path)
            # For debug output
            if self.config.homog.debug:
                self.images = (ref_img, frag_img)

            feats_frag = self.extractor.extract(frag_img.to(self.device))
            # Find matches between images
            matches_a_b = self.matcher({"image0": feats_ref, "image1": feats_frag})
            # Ger homography from image b to a
            H, m, mkpts = self.get_homography(feats_ref, feats_frag, matches_a_b, (0, idx))
            homographies.append(H)

            corrs.append(mkpts)

        return homographies, corrs


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


        if len(np.asarray(m_kpts2.cpu())) < 10 or len(np.asarray(m_kpts1.cpu())) < 10:
            print("Not enough points")
            return

        H, mask = cv.findHomography(np.asarray(m_kpts2.cpu()), np.asarray(m_kpts1.cpu()), cv.RANSAC, 5.0)
        return H, mask, (np.asarray(m_kpts2.cpu()), np.asarray(m_kpts1.cpu()))
