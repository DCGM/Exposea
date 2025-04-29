import os
import hydra
import pickle
import torch.cuda
import datetime
import logging
import diskcache
import tqdm

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
            self.cache = diskcache.Cache(self.config.cache.path, timeout=60*60, cull_limit=0, eviction_policy="none")
            self.cache.clear()
            self.cache.max_size = 1024 * 1024 * self.config.cache.size_mb
        else:
            self.cache = None

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
        self.ref_path, self.frag_paths = self.load_image_paths(True)

        # debug printouts
        self.debug = self.config.debug

    def __del__(self):
        self.cache.clear()

    def run(self):
        """
        Runs the main stitch app.
        """
        if self.debug:
            print("Torch cuda", torch.cuda.is_available())

        # Make sure that the overview image is in final resolution
        # TODO Change this
        self.resize_reference()
        # TODO CHECK IMG TYPES ALLWAYS FLOATS
        # TODO GLIMUR
        # Estimate homographies
        logging.info("Estimating homographies")
        homographies = self.run_homog()

        # Apply the homography
        logging.info("Warping images with estimated homography")
        warped_images = self.stitcher.warp_images(homographies, self.frag_paths, self.cache)

        # Stitch the images for debug output
        if self.debug:
            stitched = self.stitcher.blend("weighted", args={"fragments":warped_images, "cache": self.cache})
            cv.imwrite("./plots/homog_stitch.jpg", stitched)

        # Warp images with optical flow
        logging.info("Estimating optical flow")
        flow_warped_images = self.run_flow(self.ref_path, warped_images)
        # Del due to memory consumption
        del warped_images

        # Stitch the images for debug output
        if self.debug:
            stitched = self.stitcher.blend("weighted", args={"fragments":flow_warped_images, "cache": self.cache})
            cv.imwrite("./plots/flow_stitch.jpg", stitched)


        logging.info("Adjusting light")
        light_adjusted = self.run_light_equal(self.ref_path, flow_warped_images)
        # Del due to memory consumption
        del flow_warped_images

        logging.info("Stitching actual image")
        # Flow Stitch
        stitched = self.stitcher.blend(self.config.stitcher.mode, args={"imgs":light_adjusted, "Hs":homographies, 'cache':self.cache})
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
            homographies, _ = self.homog_estimator.new_register(self.ref_path, self.frag_paths)
            if self.config.homog.save:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                with open(f"cache/homogs/opt_hom_{timestamp}.pkl", "wb") as f:
                    pickle.dump(homographies, f)

        return homographies

    def run_flow(self, ref_path, warped_fragments):
        """
        Runs the flow estimation
        Args:
            warped_images (dict): Dict of homography warped images {name: (image, mask)}

        Returns:
        """
        # Store or load flows
        if self.config.optical.load:
            with open(self.config.optical.load, "rb") as f:
                saved_flows = pickle.load(f)
                flow_warped_images = []
                if self.config.cache.is_on:
                    for key, val in enumerate(saved_flows):
                        self.cache[key] = val
                        flow_warped_images.append(key)
                else:
                    flow_warped_images = saved_flows

        else:
            ref_img = cv.imread(ref_path)
            flow_warped_images = self.run_optical_flow(ref_img, warped_fragments)
            # Save flows if needed
            if self.config.homog.save:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                with open(f"cache/flows/flows_{timestamp}.pkl", "wb") as f:
                    if self.cache is not None:
                        collect = {}
                        for key in flow_warped_images:
                            collect[key] = self.cache[key]
                        pickle.dump(collect, f)
                    else:
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
        ref_img = cv.imread(ref)
        light_adjusted = equalize(flow_warped_images, self.cache, ref_img, config=self.config)

        return light_adjusted


    def resize_reference(self):
        """
        Resizes the reference image to final resolution
        Returns:
        """
        ref = cv.imread(self.ref_path)
        h, w =  self.config.data.final_res
        ref = cv.resize(ref, (w,h), interpolation=cv.INTER_CUBIC)
        cv.imwrite(self.ref_path, ref)


    def load_image_paths(self, sort):

        img_names = os.listdir(self.config.data.img_folder)
        overview_name = str(self.config.data.img_overview)
        # Get overview image and save it separately for visualization
        if overview_name in img_names:
            ref_path = os.path.join(self.config.data.img_folder, overview_name)
        else:
            raise ValueError("Overview image not found")
        if sort:
            img_names = sorted(img_names, key=lambda x: int(x.split('.')[0]))

        # Save only the paths as we need to load the images in different formats
        # for visualization and homography
        frag_path = []
        for name in img_names:
            if name != overview_name:
                img_p = os.path.join(self.config.data.img_folder, name)
                frag_path.append(img_p)
        return ref_path, frag_path

    def run_optical_flow(self, ref_img, warp_frags):

        img_pbar = tqdm.tqdm(total=len(warp_frags), desc='Processing fragments using optical flow',
                        position=0, leave=False, ncols=100, colour='green')

        flow_fragments = []
        for key, val in enumerate(warp_frags):
            if self.cache is not None:
                frag, mask = self.cache.pop(val)
            else:
                frag, mask = val

            flow = self.optical.estimate_flow(ref_img, frag)
            # Apply image flows
            flow_frag = self.optical.warp_image(frag, flow)
            # Save debug img
            if self.config.optical.debug:
                cv.imwrite(f"./plots/before_warp_{key}.jpg", frag)
                cv.imwrite(f"./plots/warped_flow_{key}.jpg", flow_frag)
            # Cache 2
            if self.cache is not None:
                cache_key = f"flow_frag_{key}"
                with open("cache/test_pkl.pkl", "wb") as f:
                    pickle.dump((flow_frag, mask), f)
                self.cache.add(cache_key, (flow_frag, mask))
                flow_fragments.append(cache_key)
            else:
                flow_fragments.append((flow_frag, mask))
            img_pbar.update(1)

        return flow_fragments


    def compose_final_img(self, warped_flows, overview):
        pass


# Launch the application for stitching the image
@hydra.main(version_base=None, config_path="configs", config_name="david")
def main(config):
    app = StitchApp(config)
    app.run()

if __name__ == "__main__":
    main()





