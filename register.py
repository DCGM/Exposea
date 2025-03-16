from multiprocessing.util import debug

import cv2 as cv
import os
import hydra
import numpy as np
import pickle

import torch.cuda
import tqdm
import copy
import datetime

from HomogEst import HomogEstimator
from Stitcher import Stitcher
from Optical import OpticalFlow

class CustomError(Exception):
    """Custom exception with a message."""
    def __init__(self, message):
        super().__init__(message)

class StichApp():

    def __init__(self, config):
        # File with all the configurations
        self.config = config
        self.pairs = config.data.img_pairs
        self.debug = config.debug
        # Initialize homography estimator
        self.homog_estimator = None
        self.init_homog_est()
        # Load images paths
        self.load_image_paths(True)
        self.stitcher = Stitcher(config, debug=True)
        # Optical flow initialization
        self.optical = OpticalFlow(config)


    def init_homog_est(self):
        if self.config.homog.type == "default":
            self.homog_estimator = HomogEstimator(self.config)
        else:
            raise CustomError("Homography estimator type not implemented. Available types: default")

    def resize_reference(self):
        p = os.path.join(str(self.config.data.img_folder), str(self.config.data.img_overview))
        ref = cv.imread(p)
        h, w =  self.config.data.final_res
        ref = cv.resize(ref, (w,h), interpolation=cv.INTER_CUBIC)
        cv.imwrite(p, ref)

    def run(self):
        """
        Runs the stitching function in correct order
        :return:
        """
        print("Torch cuda", torch.cuda.is_available())

        # Make sure that the overview image is in final resolution
        # TODO Change this
        self.resize_reference()

        print("Estimating homographs")
        # Load or estimate homographies
        if self.config.homog.load_homog:
            with open(self.config.homog.load_homog, "rb") as f:
                homographies = pickle.load(f)
        else:
            homographies = self.homog_estimator.register(self.img_paths)

        # Apply the homography
        print("Warping and saving images with estimated homography")
        warped_images = self.stitcher.warp_images(homographies, self.img_paths, self.pairs)
        # Stitch the images for debug output
        stitched = self.stitcher.blend("weighted", args=warped_images)
        cv.imwrite("./plots/final_stitch.jpg", stitched)
        # Warp images with optical flow
        # Store or load flows for debug

        if self.config.optical.load_flows:
            with open(self.config.optical.load_flows, "rb") as f:
                flow_warped_images = pickle.load(f)
        else:
            flow_warped_images = self.run_optical_flow(warped_images)
            if self.config.debug:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                with open(f"flows_{timestamp}.pkl", "wb") as f:
                    pickle.dump(flow_warped_images, f)


        # Flow Stitch
        stitched = self.stitcher.blend(self.config.stitch_type, args={"imgs":flow_warped_images, "Hs":homographies })
        cv.imwrite("./plots/final_flow_stitch.jpg", stitched)

    def load_image_paths(self, sort):

        img_names = os.listdir(self.config.data.img_folder)
        overview_name = str(self.config.data.img_overview)
        # Get overview image and save it separately for visualization
        if overview_name in img_names:
            ov_img_p = os.path.join(self.config.data.img_folder, overview_name)
            self.ov_img_cv = cv.imread(ov_img_p)
        else:
            raise CustomError("Overview image not found")
        if sort:
            img_names = sorted(img_names, key=lambda x: int(x.split('.')[0]))

        # Save only the paths as we need to load the images in different formats
        # for visualization and homography
        self.img_paths = []
        for name in img_names:
            img_p = os.path.join(self.config.data.img_folder, name)
            self.img_paths.append(img_p)

    def run_optical_flow(self, homog_images):

        img_flows = self.optical.estimate_flows(homog_images)
        # Apply image flows
        flow_warped_images = {}
        for flow, (key, val) in zip(img_flows, homog_images.items()):

            warped = self.optical.warp_image(val[0], flow)
            mask = (warped > 0)
            cv.imwrite(f"./plots/before_warp_{key}.jpg", val[0])
            cv.imwrite(f"./plots/warped_flow_{key}.jpg", warped)
            flow_warped_images[key] = (warped, mask)

        return flow_warped_images


    def compose_final_img(self, warped_flows, overview):
        pass


# Lauch the application for stitching the image
@hydra.main(version_base=None, config_path="configs", config_name="cave1")
def main(config):
    app = StichApp(config)
    app.run()

if __name__ == "__main__":
    main()





