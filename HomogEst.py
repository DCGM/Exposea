from lightglue.utils import load_image, rbd
import cv2 as cv
import os
from lightglue import viz2d
from marek.homography_optimalizer import *
from lightglue import LightGlue, SuperPoint
import torch
from omegaconf import OmegaConf

def load_imgs(img_paths):
    # Load images in super point and light glue format
    imgs = {}
    for p in img_paths:
        _, filename = os.path.split(p)
        filename, _ = filename.split(".")
        img = load_image(p)
        imgs[int(filename)] = img
    return imgs

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
        self.pairs = OmegaConf.to_container(config.data.img_pairs)
        self.images = None

    def register(self, img_paths: np.ndarray):
        """
        Main function that handles registration
        Args:
            img_paths:
        Returns:

        """
        # Load images
        self.images = load_imgs(img_paths)
        # Register images by pair returns homography and corresponding points for each pair
        homographies, corrs = self.register_pairs()
        # Normalize homographies into common coordinate frame
        homographies = self.normalize_homographies(homographies)

        # Global optimization
        if self.config.homog.do_optimization:
            # Prepare input for optimizer
            correspondences = []
            for idx, pair in enumerate(self.pairs):
                correspondences.append({"pair": pair, "points": corrs[idx]})
            hom_list = []
            # Remove identity homography from list
            for key, value in homographies.items():
                if key == 0:
                    continue
                hom_list.append(value)
            # Run optimization
            homographies = self.homog_opt.optimize(self.pairs, hom_list, correspondences)
            # Save optimized homographies
        return homographies

    def match_feat_pairs(self, feats1, feats2):
        matches = self.matcher({"image0": feats1, "image1": feats2})
        return matches

    def register_pairs(self):
        homographies = []
        corrs = []
        # Get homography for each pair
        for idx, pair in enumerate(self.pairs):
            # TODO FIX
            fragment_a = self.images[pair[0]]
            fragment_b = self.images[pair[1]]
            # Extract features from both images
            feats_a = self.extractor.extract(fragment_a.to(self.device))
            feats_b = self.extractor.extract(fragment_b.to(self.device))
            # Find matches between images
            matches_a_b = self.matcher({"image0": feats_a, "image1": feats_b})
            # Ger homography from image b to a
            H, m, mkpts = self.get_homography(feats_a, feats_b, matches_a_b, (pair[0], pair[1]))

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
            _save_debug_imgs([self.images[pair[0]], m_kpts1, kpts1],
                             [self.images[pair[1]], m_kpts2, kpts2],
                             matches12,
                             path=f"./plots/matches_{pair[0]}_{pair[1]}.jpeg")
            _save_key_imgs([self.images[pair[0]], m_kpts1, kpts1],
                            [self.images[pair[1]], m_kpts2, kpts2],
                            path=f"./plots/kpts_{pair[0]}_{pair[1]}.jpeg")


        if len(np.asarray(m_kpts2.cpu())) < 10 or len(np.asarray(m_kpts1.cpu())) < 10:
            print("Not enough points")
            return

        H, mask = cv.findHomography(np.asarray(m_kpts2.cpu()), np.asarray(m_kpts1.cpu()), cv.RANSAC, 5.0)
        return H, mask, (np.asarray(m_kpts2.cpu()), np.asarray(m_kpts1.cpu()))


    def normalize_homographies(self, homographies, ref_idx=0):

        global_homographies = {ref_idx: np.eye(3)}

        for idx, (dst, src) in enumerate(self.pairs):
            if dst == ref_idx:
                global_homographies[idx + 1] = homographies[idx]
                continue
            if src == ref_idx:
                global_homographies[idx + 1] = np.linalg.inv(homographies[idx])
                continue

        for idx, (dst, src) in enumerate(self.pairs):
            if dst == ref_idx or src == ref_idx:
                continue

            if dst in global_homographies.keys():
                global_homographies[idx + 1] = global_homographies[dst] @ homographies[idx]

            elif src in global_homographies.keys():
                global_homographies[idx + 1] = global_homographies[src] @ np.linalg.inv(homographies[idx])
        return global_homographies
