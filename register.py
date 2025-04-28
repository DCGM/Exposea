import os
import hydra
import pickle
import torch.cuda
import datetime
import logging
import diskcache

from HomogEst import HomogEstimator
from Stitcher import Stitcher
from Optical import OpticalFlow
from LightEqual import *


class StitchApp():

    def __init__(self, config):
        # Config file
        self.config = config
        # Init cache dir
        if self.config.cache.is_on:
            self.cache = diskcache.Cache(self.config.cache.cache_path)
            self.cache.max_size = 1024 * 1024 * self.config.cache.max_size

        self.pairs = config.data.img_pairs


        # TODO Change
        # Initialize homography estimator
        if self.config.homog.type == "default":
            self.homog_estimator = HomogEstimator(self.config)
        else:
            raise ValueError("Homography estimator type not implemented. Available types: default")

        # Optical flow initialization
        self.optical = OpticalFlow(config)
        # Main stitcher
        self.stitcher = Stitcher(config, debug=True)

        # Load images paths
        self.load_image_paths(True)

        # debug printouts
        self.debug = self.config.debug

    def run(self):
        """
        Runs the main stitch app.
        """
        if self.debug:
            print("Torch cuda", torch.cuda.is_available())

        # Make sure that the overview image is in final resolution
        # TODO Change this
        self.resize_reference()

        # Estimate homographies
        logging.info("Estimating homographies")
        homographies = self.run_homog()

        # Apply the homography
        logging.info("Warping images with estimated homography")
        warped_images = self.stitcher.warp_images(homographies, self.img_paths, self.pairs)

        # Stitch the images for debug output
        if self.debug:
            stitched = self.stitcher.blend("weighted", args=warped_images)
            cv.imwrite("./plots/homog_stitch.jpg", stitched)

        # Warp images with optical flow
        logging.info("Estimating optical flow")
        ref = warped_images[0]
        flow_warped_images = self.run_flow(warped_images)
        # Del due to memory consumption
        del warped_images

        # # Stitch the images for debug output
        if self.debug:
            stitched = self.stitcher.blend("weighted", args=flow_warped_images)
            cv.imwrite("./plots/flow_stitch.jpg", stitched)


        logging.info("Adjusting light")
        light_adjusted = flow_warped_images #self.run_light_equal(ref, flow_warped_images)
        # Del due to memory consumption
        del flow_warped_images

        logging.info("Stitching actual image")
        # Flow Stitch
        stitched = self.stitcher.blend(self.config.stitcher.mode, args={"imgs":light_adjusted, "Hs":homographies })
        cv.imwrite("./plots/final_flow_stitch.jpg", stitched)

    def run_homog(self):
        """
        Runs the homography estimation
        If save in config it saves the homographies
        Returns:

        """
        # Load or estimate homographies
        if self.config.homog.load:
            with open(self.config.homog.load, "rb") as f:
                homographies = pickle.load(f)
        else:
            # The img paths is sent to load the images in correct format for feature extraction and matching
            homographies = self.homog_estimator.register(self.img_paths)
            if self.config.homog.save:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                with open(f"opt_hom_{timestamp}.pkl", "wb") as f:
                    pickle.dump(homographies, f)

        return homographies

    def run_flow(self, warped_images):
        """
        Runs the flow estimation
        Args:
            warped_images (dict): Dict of homography warped images {name: (image, mask)}

        Returns:
        """
        # Store or load flows
        if self.config.optical.load:
            with open(self.config.optical.load, "rb") as f:
                flow_warped_images = pickle.load(f)
        else:
            flow_warped_images = self.run_optical_flow(warped_images)
            # Save flows if needed
            if self.config.homog.save:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                with open(f"flows_{timestamp}.pkl", "wb") as f:
                    pickle.dump(flow_warped_images, f)

        return flow_warped_images

    def run_light_equal(self, ref, flow_warped_images):
        """
        Runs the flow estimation
        Args:
            ref (tuple): Reference image and its mask
            flow_warped_images (dict): Dict of flow warped images {name: (image, mask)}

        Returns:
        """
        # Equalize light
        flow_warped_images[0] = ref
        light_adjusted = equalize(flow_warped_images, config=self.config)
        light_adjusted[0] = ref

        return light_adjusted


    def resize_reference(self):
        """
        Resizes the reference image to final resolution
        Returns:
        """
        p = os.path.join(str(self.config.data.img_folder), str(self.config.data.img_overview))
        ref = cv.imread(p)
        h, w =  self.config.data.final_res
        ref = cv.resize(ref, (w,h), interpolation=cv.INTER_CUBIC)
        cv.imwrite(p, ref)


    def load_image_paths(self, sort):

        img_names = os.listdir(self.config.data.img_folder)
        overview_name = str(self.config.data.img_overview)
        # Get overview image and save it separately for visualization
        if overview_name in img_names:
            ov_img_p = os.path.join(self.config.data.img_folder, overview_name)
            self.ov_img_cv = cv.imread(ov_img_p)
        else:
            raise ValueError("Overview image not found")
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


# Launch the application for stitching the image
@hydra.main(version_base=None, config_path="configs", config_name="debug")
def main(config):
    app = StitchApp(config)
    app.run()

if __name__ == "__main__":
    main()





