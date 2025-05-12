import os
import os.path as osp
import hydra
import pickle
import torch.cuda
import datetime
import logging
import tqdm

from HomogEst import HomogEstimator
from Stitcher import Stitcher, ActualBlender
from Optical import OpticalFlow
from LightEqual import *


class StitchApp():

    def __init__(self, config):
        # Config file
        self.config = config
        # Init cache dir

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
        # Memory rewrite
        # Estimate homographies
        logging.info("Estimating homographies")
        homographies = self.run_homog()
        # Initialize progressive blender of fragments
        prog_blend = ActualBlender(self.config)

        for f_idx, frag_path in enumerate(self.frag_paths):
            # Apply homography
            homog_frag = homographies[f_idx]

            # Apply the homography
            logging.info("Warping images with estimated homography")
            warped_fragment, frag_mask = self.stitcher.warp_image(homog_frag, frag_path)

            # Debug output
            if self.debug:
                cv.imwrite(f"./plots/homog_{f_idx}.jpg", warped_fragment)

            # Warp fragment with optical flow
            logging.info("Estimating optical flow")
            flow_fragment = self.run_flow(self.ref_path, warped_fragment, osp.basename(frag_path))

            # Images for debug output
            if self.debug:
                cv.imwrite(f"./plots/flow_{f_idx}.jpg", flow_fragment)

            logging.info("Adjusting light")
            light_adjusted = self.run_light_equal(self.ref_path, flow_fragment, frag_mask)

            logging.info("Adding fragment to final blend")
            prog_blend.add_fragment(light_adjusted, frag_mask, homog_frag, f_idx)


        cv.imwrite("./plots/final_flow_stitch.jpg", prog_blend.get_current_blend())

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
            homographies, _ = self.homog_estimator.register(self.ref_path, self.frag_paths)
            if self.config.homog.save:
                os.makedirs(self.config.homog.save, exist_ok=True)
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                with open(f"cache/homogs/opt_hom_{timestamp}.pkl", "wb") as f:
                    pickle.dump(homographies, f)

        return homographies

    def run_flow(self, ref_path, warped_frag, frag_name):
        """
        Gets optical flow, either loaded or calculated
        Args:
           ref_path:
           warped_images:
           frag_name:

        Returns:

        """
        # Check if load optical else compute flow
        if self.config.optical.load:
            # Check if path exist else compute flow
            path = osp.join(self.config.optical.load, f'flow_{self.config.exp_name}_{frag_name}.npy')
            if osp.exists(path):
                with open(path, "rb") as f:
                    flow = np.load(f)
            else:
                print(f"Optical flow {frag_name} not found")
                # Get ref image and compute flow
                ref_img = cv.imread(ref_path)
                flow = self.optical.estimate_flow(ref_img, warped_frag)
        else:
            # Get ref image and compute flow
            ref_img = cv.imread(ref_path)
            flow = self.optical.estimate_flow(ref_img, warped_frag)
        # If save path specified save flows
        if self.config.optical.save:
            os.makedirs(self.config.optical.save, exist_ok=True)
            path = osp.join(self.config.optical.save, f'flow_{self.config.exp_name}_{frag_name}.npy')
            with open(path, "wb") as f:
                np.save(f, flow)
        flow_frag = self.optical.warp_image(warped_frag, flow)
        return flow_frag



    def run_light_equal(self, ref_path, flow_fragment, frag_mask):
        """
        Runs the flow estimation
        Args:
            ref (tuple): Reference image and its mask
            flow_warped_images (dict): Dict of flow warped images {name: (image, mask)}

        Returns:
        """
        # Equalize light
        ref_img = cv.imread(ref_path)
        light_adjusted = equalize_frag(flow_fragment, frag_mask.copy(), ref_img, config=self.config)

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


def create_dirs():
    os.makedirs("plots", exist_ok=True)
    os.makedirs("cache", exist_ok=True)
    os.makedirs("cache/homogs", exist_ok=True)
    os.makedirs("cache/flows", exist_ok=True)
# Launch the application for stitching the image
@hydra.main(version_base=None, config_path="configs", config_name="debug")
def main(config):
    app = StitchApp(config)
    app.run()

if __name__ == "__main__":
    main()





