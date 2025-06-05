import logging
import os
import os.path as osp

import hydra
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra

import pickle
import torch.cuda
import datetime

from omegaconf import OmegaConf

from modules.HomogEst import HomogEstimator
from modules.Stitcher import Stitcher, ActualBlender, DebugBlender
from modules.Optical import OpticalFlow
from modules.LightEqual import *

# For profile only
from utils import timer

class StitchApp():

    def __init__(self, config):
        # Config file
        self.config = config
        # Init cache dir

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

        # Timer
        self.run_timer, self.flow_timer, self.lo_timer = timer.Timer(), timer.Timer(), timer.Timer()


    def run(self):
        """
        Runs the main stitch app.
        """

        if self.debug:
            print("Torch cuda", torch.cuda.is_available())

        self.run_timer.tic()
        # Make sure that the overview image is in compact resolution
        self.resize_reference(self.config.final_res)
        # TODO CHECK IMG TYPES ALLWAYS FLOATS
        # TODO GLIMUR
        # Memory rewrite
        # Estimate homographies
        logging.info("Estimating homographies")
        homographies = self.run_homog(resize=False)



        # Initialize progressive blender of fragments
        homog_blender = DebugBlender(self.config.final_res, self.config)

        prog_blend = ActualBlender(self.config)
        stitch_progress = tqdm.tqdm(total=len(self.frag_paths), leave=True ,desc='Stitching images ', position=1, ncols=100, colour='blue')
        for f_idx, frag_path in enumerate(self.frag_paths):
            torch.cuda.reset_peak_memory_stats()
            self.debug_idx = f_idx
            # Apply homography
            homog_frag = homographies[f_idx]

            # Apply the homography
            logging.info("Warping images with estimated homography")
            warped_fragment, frag_mask = self.stitcher.warp_image(homog_frag, frag_path)

            # Debug output
            if self.debug:
                homog_blender.add_fragment(warped_fragment, frag_mask)
                cv.imwrite(f"./plots/homog_{f_idx}.jpg", homog_blender.get_current_blend())

            # Warp fragment with optical flow
            logging.info("Estimating optical flow")
            flow_fragment = self.run_flow(self.ref_path, warped_fragment, osp.basename(frag_path))
            # Memory clean
            torch.cuda.empty_cache()

            # Images for debug output
            if self.debug:
                cv.imwrite(f"./plots/flow_{f_idx}.jpg", flow_fragment)

            logging.info("Adjusting light")
            light_adjusted = self.run_light_equal(self.ref_path, flow_fragment, frag_mask)

            logging.info("Adding fragment to final blend")
            prog_blend.add_fragment(light_adjusted, frag_mask, homog_frag, f_idx)

            peak = torch.cuda.max_memory_allocated()
            logging.info(f"Peak usage: {peak / 1024 ** 2:.2f} MB")

            # Memory clean
            torch.cuda.empty_cache()
            stitch_progress.update(1)

        final_img = prog_blend.get_current_blend()
        # glymur.Jp2k("./plots/final_stitch_jp2k.jp2", data=final_img)
        cv.imwrite("./plots/final_stitch.jpg", final_img)

        logging.info(f"Time | Optical flow {self.flow_timer.average_time}")
        logging.info(f"Time | Light optim {self.lo_timer.average_time}")
        logging.info(f"Time | Finished stitching {self.run_timer.toc(False)}")


    def run_homog(self, resize=False):
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

        # Resize the homography to correct scale
        if resize:
            scale = self.config.final_res[0] / self.config.process_res[0]
            scaled_homographies = []
            for h  in homographies:
                D = np.array([[scale, 0, 0],
                              [0, scale, 0],
                              [0, 0, 1]])
                h_scaled = D @ h
                scaled_homographies.append(h_scaled)

            return scaled_homographies
        else:
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
        self.flow_timer.tic()
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
                flow = self.optical.estimate_flow(ref_img, warped_frag, self.debug_idx)
        else:
            # Get ref image and compute flow
            ref_img = cv.imread(ref_path)
            flow = self.optical.estimate_flow(ref_img, warped_frag,  self.debug_idx)
        # If save path specified save flows
        if self.config.optical.save:
            os.makedirs(self.config.optical.save, exist_ok=True)
            path = osp.join(self.config.optical.save, f'flow_{self.config.exp_name}_{frag_name}.npy')
            with open(path, "wb") as f:
                np.save(f, flow)
        flow_frag = self.optical.warp_image(warped_frag, flow)
        self.flow_timer.toc()
        return flow_frag



    def run_light_equal(self, ref_path, flow_fragment, frag_mask):
        """
        Runs the flow estimation
        Args:
            ref (tuple): Reference image and its mask
            flow_warped_images (dict): Dict of flow warped images {name: (image, mask)}

        Returns:
        """
        self.lo_timer.tic()
        # Equalize light
        ref_img = cv.imread(ref_path)
        # Tile the image for memory consumption
        if self.config.light_optim.use_tile:
            light_adjusted = tile_equalize_fragments(flow_fragment, frag_mask.copy(), ref_img, config=self.config)
        else:
            light_adjusted = equalize_frag(flow_fragment, frag_mask.copy(), ref_img, config=self.config)

        self.lo_timer.toc()
        return light_adjusted


    def resize_reference(self, size):
        """
        Resizes the reference image to final resolution
        Returns:
        """
        ref = cv.imread(self.ref_path)
        h, w = size
        ref = cv.resize(ref, (w,h), interpolation=cv.INTER_CUBIC)
        self.ref_path = "cache/ref_resized.png"
        cv.imwrite(self.ref_path, ref)


    def load_image_paths(self, sort):

        img_names = os.listdir(self.config.input_folder)
        overview_name = str(self.config.img_overview)
        # Get overview image and save it separately for visualization
        if overview_name in img_names:
            ref_path = os.path.join(self.config.input_folder, overview_name)
        else:
            raise ValueError("Overview image not found")
        if sort:
            img_names = sorted(img_names, key=lambda x: int(x.split('.')[0]))

        # Save only the paths as we need to load the images in different formats
        # for visualization and homography
        frag_path = []
        for name in img_names:
            if name != overview_name:
                img_p = os.path.join(self.config.input_folder, name)
                frag_path.append(img_p)
        return ref_path, frag_path


def create_dirs():
    os.makedirs("plots", exist_ok=True)
    os.makedirs("cache", exist_ok=True)
    os.makedirs("cache/homogs", exist_ok=True)
    os.makedirs("cache/flows", exist_ok=True)


def compose_configs(input_cfg):
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    logging.info("Loading required arguments")
    with initialize(config_path='configs/presets/', version_base=None):
        default_cfg = compose(config_name='default')
        preset_cfg = compose(input_cfg.preset_name)

    logging.info("Checking required arguments")

    for req_args in list(default_cfg.app.required_args):
        if req_args not in input_cfg.keys():
            logging.error(f"Required argument {req_args} is not defined")
            raise ValueError(f"Required argument {req_args} is missing")

    logging.info("Composing configs")
    # Merge them manually: later ones overwrite earlier ones
    config = OmegaConf.merge(default_cfg, preset_cfg, input_cfg)

    return config



# Launch the application for stitching the image
@hydra.main(version_base=None, config_path="configs", config_name="in_cave1")
def main(config):
    # TODO Change
    # Configure logging
    logging.basicConfig(
        filename="app_log.txt",
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    # Crete caching dirs
    create_dirs()
    # Merge configs, allows for specification of hand picked parameters
    config = compose_configs(config)
# Run main stitcher
    app = StitchApp(config)
    app.run()

if __name__ == "__main__":
    main()





