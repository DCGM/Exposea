import logging
import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = '10000000000'
import os.path as osp
import argparse
import shutil
import subprocess

import pprint
import pickle

import cv2
import numpy as np
import torch.cuda
import datetime

from omegaconf import OmegaConf

from modules.HomogEst import HomogEstimator
from modules.Stitcher import Stitcher, ActualBlender, DebugBlender
from modules.Optical import OpticalFlow
from modules.LightEqual import *
from utils.rectangularize import clip, order_points
from utils.utils import scale_homog

import gc

# For profile only
from utils import timer

class StitchApp():

    def __init__(self, config):
        self.logger = logging.getLogger('STITCHER')
        self.logger.info('Initializing STITCHER')
        self.logger.info("Config:\n%s", pprint.pformat(config))
        # Config file
        self.config = config
        # Output
        self.out_dir = config.output_folder
        self.img_dir = osp.join(config.input_folder, 'images')

        if self.config.homog.type == "default":
            self.homog_estimator = HomogEstimator(self.config)
        else:
            self.logger.error("Unknown homog type: %s", self.config.homog.type)
            raise ValueError("Homography estimator type not implemented. Available types: default")

        # Optical flow initialization
        self.optical = OpticalFlow(config)
        # Main stitcher
        self.stitcher = Stitcher(config, debug=True)

        # Load images paths
        self.ref_path, self.frag_paths = self.load_image_paths(False)
        # Place holder for resized reference
        self.ref_resized_path = self.ref_path

        # debug printouts
        self.debug = self.config.debug

        # Timer
        self.run_timer, self.flow_timer, self.lo_timer = timer.Timer(), timer.Timer(), timer.Timer()


    def run(self):
        """
        Runs the main stitch app.
        """

        if self.debug:
            self.logger.info("Torch cuda %s", torch.cuda.is_available())

        self.run_timer.tic()
        # Reference image has to be rectangularized and resized to final resolution
        self.rect_ref()
        # Calculates nad cache resized reference to process resolution for computation
        self.calc_process_params()

        # Estimate homographies

        self.logger.info(f"Estimating homographies for {len(self.frag_paths)} images")
        homographies = self.run_homog(resize=False)

        # Initialize progressive blender of fragments
        if self.debug:
            homog_blender = DebugBlender(self.process_HW, self.config)

        self.logger.info("Start of sequential image stitching")
        prog_blend = ActualBlender(self.config, cv.imread(self.ref_path))
        stitch_progress = tqdm.tqdm(total=len(self.frag_paths), leave=True ,desc='Stitching images ', position=1, ncols=100, colour='blue')
        for f_idx, frag_path in enumerate(self.frag_paths):

            torch.cuda.reset_peak_memory_stats()
            self.debug_idx = f_idx

            # Processing in process resolution
            #######################################
            # Apply the homography
            homog_frag = homographies[f_idx]
            self.logger.info(f"[{f_idx}]    Warping with estimated homography")
            warped_fragment, frag_mask = self.stitcher.warp_image(homog_frag, frag_path, res=self.process_HW)

            # Debug output
            if self.debug:
                homog_blender.add_fragment(warped_fragment, frag_mask)
                cv.imwrite(f"./plots/homog_{f_idx}.jpg", homog_blender.get_current_blend())

            # Estimate optical flow
            _, flow = self.run_flow(self.ref_resized_path, warped_fragment, osp.basename(frag_path))

            # Processing in final resolution
            #######################################
            self.logger.info(f"[{f_idx}]  Scaling flow by {self.final_scale}")
            homog_frag = scale_homog(homog_frag, self.final_scale)
            flow *= self.final_scale
            flow = np.array(cv.resize(flow, (self.config.final_res[1],self.config.final_res[0]), cv.INTER_LINEAR), dtype=np.float16)
            warped_fragment, frag_mask = self.stitcher.warp_image(homog_frag, frag_path)
            # Images for debug output
            if self.debug:
                cv.imwrite(f"./plots/warped_{f_idx}.jpg", warped_fragment)

            flow_fragment = self.optical.warp_image(warped_fragment, flow)
            frag_mask = np.astype(self.optical.warp_mask(np.astype(frag_mask, np.float32), flow), bool)
            del flow, warped_fragment

            # Warp fragment with optical flow
            self.logger.info(f"[{f_idx}]    Estimating optical flow")

            # Memory clean
            torch.cuda.empty_cache()

            # Images for debug output
            if self.debug:
                cv.imwrite(f"./plots/flow_{f_idx}.jpg", flow_fragment)

            self.logger.info(f"[{f_idx}] Adjusting light")
            light_adjusted, _ = self.run_light_equal(self.ref_path, flow_fragment, frag_mask, resize=False)
            del flow_fragment

            self.logger.info(f"Fragment {f_idx} adding to final blend")
            prog_blend.add_fragment(light_adjusted, frag_mask, homog_frag, f_idx)

            if self.debug:
                peak = torch.cuda.max_memory_allocated()
                self.logger.info(f"Peak usage: {peak / 1024 ** 2:.2f} MB")

            # Memory clean
            if f_idx % 4 == 0:
                torch.cuda.empty_cache()
                gc.collect()

            stitch_progress.update(1)

        final_img = prog_blend.get_current_blend()


        self.save_final_img(final_img)

        self.logger.info(f"Average Time | Optical flow {self.flow_timer.average_time}")
        self.logger.info(f"Average Time | Light optim {self.lo_timer.average_time}")
        self.logger.info(f"Average Time | Finished stitching {self.run_timer.toc(False)}")



    def save_final_img(self, img):

        if hasattr(self.config, 'save_format') and  self.config.save_format in ['jp2', 'j2k']:
            cv.imwrite(osp.join(self.out_dir, "final_stitch.png"), img)
            save_name = f"final_stitch.{self.config.save_format}"
            self.save_in_jp2(osp.join(self.out_dir, "final_stitch.png"), osp.join(self.out_dir, save_name))
        elif hasattr(self.config, 'save_format') and  self.config.save_format in ['tiff', 'tif']:
            save_name = f"final_stitch.{self.config.save_format}"
            cv.imwrite(osp.join(self.out_dir, save_name), img)

        else:
            cv.imwrite(osp.join(self.out_dir, "final_stitch.png"), img)
            save_name = f"final_stitch.{self.config.save_format}"
            self.save_in_jp2(osp.join(self.out_dir, "final_stitch.png"), osp.join(self.out_dir, save_name))
            raise NotImplementedError('Supported extensions: tif, tiff, jp2, j2k. Imaged saved as jp2')


    def save_in_jp2(self, i_path, o_pth):
        self.logger.info(f"Saving {o_pth} using jp2 format")
        cmd = [
            'opj_compress',
            '-i', i_path,
            '-o', o_pth,
            '-t', '4069,4096',
            '-p', 'RPCL',
            '-r', '1',
            '-c', '[256,256]',
            '-TLM',
            '-M', '1',
            '-SOP',
            '-EPH'
        ]

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error: opj_compress failed with exit code {e.returncode}")
            raise

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
            homographies, _ , _ = self.homog_estimator.register(self.ref_resized_path, self.frag_paths)
           # self.frag_paths = [val for idx, val in enumerate(self.frag_paths) if idx not in to_del]

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

    def run_flow(self, ref_path, warped_frag, frag_name, resize=False):
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
        # flow_frag = self.optical.warp_image(warped_frag, flow)
        self.flow_timer.toc()
       # return flow_frag, flow
        return None, flow


    def run_light_equal(self, ref_path, flow_fragment, frag_mask, resize=False):
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
        if resize:
            ref_img = cv.resize(ref_img, [self.config.final_res[1], self.config.final_res[0]], cv.INTER_AREA)
        # Tile the image for memory consumption
        if self.config.light_optim.use_tile:
            light_adjusted, _ = tile_equalize_fragments(flow_fragment, frag_mask.copy(), ref_img, config=self.config)
            self.lo_timer.toc()
            return light_adjusted, None
        else:
            light_adjusted, m = equalize_frag(flow_fragment, frag_mask.copy(), ref_img, config=self.config)
            self.lo_timer.toc()
            return light_adjusted, m

    def rect_ref(self):

        ref_img = cv.imread(self.ref_path)
        height, width = self.config.final_res
        corner_coords = list(self.config.corner_coords)
        ordered_coords = order_points(np.array(corner_coords))

        #ref_img = clip(ref_img, ordered_coords)

        pts_dst = np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
        ], dtype=np.float32)

        M = cv2.getPerspectiveTransform(ordered_coords, pts_dst)
        warped = cv2.warpPerspective(ref_img, M, (width, height), flags=cv2.INTER_CUBIC)

        path = f"./cache/ref_rect.png"
        cv.imwrite(path, warped)
        self.ref_path = path

    def calc_process_params(self):

        target_h, target_w = self.config.final_res
        target_aspect = target_w / target_h

        if target_aspect >= 1.0:
            # Wider than tall
            process_w = self.config.proc_res
            process_h = int(round(process_w / target_aspect))
        else:
            # Taller than wide
            process_h = self.config.proc_res
            process_w = int(round(process_h * target_aspect))

        self.final_scale = target_h / process_h
        self.process_HW = (process_h, process_w)
        self.resize_reference(self.process_HW)
        self.logger.info(f"Process Height, Width {self.process_HW} | Final Scale {self.final_scale}")


    def resize_reference(self, size):
        """
        Resizes the reference image to final resolution
        Returns:
        """
        ref = cv.imread(self.ref_path)
        h, w = size
        ref = cv.resize(ref, (w, h), interpolation=cv.INTER_AREA)

        path = f"./cache/ref_resized.png"
        cv.imwrite(path, ref)
        self.ref_resized_path = path

    def load_image_paths(self, sort):

        img_names = os.listdir(self.img_dir)
        ref_name = str(self.config.ref_name)
        # Get overview image and save it separately for visualization
        if ref_name in img_names:
            ref_path = os.path.join(self.img_dir, ref_name)
        else:
            raise ValueError("Overview image not found")
        if sort:
            try:
                img_names = sorted(img_names, key=lambda x: int(x.split('.')[0]))
            except ValueError:
                self.logger.warning("Fragments cannot be sorted. Continuing without sorting")
                img_names = img_names

        # Save only the paths as we need to load the images in different formats
        # for visualization and homography
        frag_path = []
        for name in img_names:
            if name != ref_name:
                img_p = os.path.join(self.img_dir, name)
                frag_path.append(img_p)
        return ref_path, frag_path



def merge_dicts(default, override):
    for k, v in override.items():
        if isinstance(v, dict) and k in default and isinstance(default[k], dict):
            merge_dicts(default[k], v)
        else:
            default[k] = v
    return default


def compose_configs(args):
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

# Launch the application for stitching the image
def main():
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
    app = StitchApp(config)
    app.run()

if __name__ == "__main__":
    main()





