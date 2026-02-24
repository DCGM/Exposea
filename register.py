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
import tifffile
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
    """
    A class for controlling the image stitching application.

    This class provides a comprehensive implementation for handling the process of image stitching, including homography
    estimation, optical flow, blending, light equalization, and saving the final results in specified formats. It uses a
    configuration object to initialize the application and manages the interaction of various components like homography
    estimators, optical flow computation, and image blending. The class logs key steps and outputs for debugging and
    progress tracking.

    Methods:
        run: Executes the main stitching process.
        save_final_img: Saves the stitched image in the specified format.
        save_in_jp2: Saves images in the JPEG 2000 format using the external tool opj_compress.
        run_homog: Performs homography estimation.
    """

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
        # Placeholder for resized reference
        self.ref_resized_path = self.ref_path

        # debug printouts
        self.debug = self.config.debug

        # Timer
        self.run_timer, self.flow_timer, self.lo_timer = timer.Timer(), timer.Timer(), timer.Timer()


    def run(self):
        """
        Runs the main image stitching process by performing the following steps sequentially:
        1. Prepares the reference image by rectangularizing and resizing it to the desired resolution.
        2. Calculates and caches parameters for image processing.
        3. Estimates homographies for a set of image fragments.
        4. Initializes progressive blending for stitching.
        5. Sequentially processes each image fragment by applying homography, estimating optical flow, adjusting lighting,
           and adding the output to the final blend.
        6. Saves the final stitched image.

        Raises:
            RuntimeError: If there are issues with image processing, stitching, or resource cleanup.

        Parameters:
            None

        Returns:
            None
        """

        if self.debug:
            self.logger.info("Torch cuda %s", torch.cuda.is_available())

        if self.config.metrics.calculate:
            os.makedirs(f"./metrics/{self.config.exp_name}", exist_ok=True)
            self.lo_frag_paths = {}

        self.run_timer.tic()
        # Reference image has to be rectangularized and resized to final resolution
        self.rect_ref()
        # Calculates nad cache resized reference to process resolution for computation
        self.calc_process_params()

        # Estimate homographies

        self.logger.info(f"Estimating homographies for {len(self.frag_paths)} images")
        homographies = self.run_homog(resize=False)
        return
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
            #_, flow = self.run_flow(self.ref_resized_path, warped_fragment, osp.basename(frag_path))

            # Processing in final resolution
            #######################################
            self.logger.info(f"[{f_idx}]  Scaling flow by {self.final_scale}")
            homog_frag = scale_homog(homog_frag, self.final_scale)
           # flow *= self.final_scale
          #  flow = np.array(cv.resize(flow, (self.config.final_res[1],self.config.final_res[0]), cv.INTER_LINEAR), dtype=np.float16)
            warped_fragment, frag_mask = self.stitcher.warp_image(homog_frag, frag_path)
            # Images for debug output
            if self.config.metrics.calculate:
                os.makedirs(f"./metrics/{self.config.exp_name}/light_adjusted", exist_ok=True)
                np.save(f"./metrics/{self.config.exp_name}/light_adjusted/frag_{f_idx}", warped_fragment)
                np.save(f"./metrics/{self.config.exp_name}/light_adjusted/mask_{f_idx}", frag_mask)
                self.lo_frag_paths[f_idx] = [f"./metrics/{self.config.exp_name}/light_adjusted/frag_{f_idx}",
                                             f"./metrics/{self.config.exp_name}/light_adjusted/mask_{f_idx}"]

            continue

            if self.debug:
                cv.imwrite(f"./plots/warped_{f_idx}.jpg", warped_fragment)

            flow_fragment = self.optical.warp_image(warped_fragment, flow)
            del warped_fragment
            gc.collect()

            print("Mask")
            frag_mask = self.optical.warp_mask(frag_mask.astype(np.float32), flow).astype(bool)
            del flow
            gc.collect()

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
            if f_idx % 1 == 0:
                torch.cuda.empty_cache()
                gc.collect()

            stitch_progress.update(1)

        final_img = prog_blend.get_current_blend()

        self.save_final_img(final_img)

        if self.config.metrics.calculate:
            np.save(f"./metrics/{self.config.exp_name}/final_img", final_img)
            np.save(f"./metrics/{self.config.exp_name}/cand_bits", prog_blend.cand_bits)
            with open(f"./metrics/{self.config.exp_name}/lo_frag_paths.pkl", "wb") as f:
                 pickle.dump(self.lo_frag_paths, f)

      #       self.cif_image_tiles(final_img, (50, 50), prog_blend.cand_bits)



        self.logger.info(f"Average Time | Optical flow {self.flow_timer.average_time}")
        self.logger.info(f"Average Time | Light optim {self.lo_timer.average_time}")
        self.logger.info(f"Average Time | Finished stitching {self.run_timer.toc(False)}")



    def cif_image_tiles(self, image, tile_size, cand_bits):
        def tile_fully_in_fragment(mask, y0, y1, x0, x1):
            """
            True if ALL pixels of the tile are inside the fragment mask.
            """
            if mask.ndim == 3:
                mask = mask[:, :, 0]

            tile = mask[y0:y1, x0:x1]
            return np.all(tile)

        H, W = image.shape[:2]
        th, tw = tile_size

        for y0 in range(0, H, th):
            y1 = min(y0 + th, H)
            for x0 in range(0, W, tw):
                x1 = min(x0 + tw, W)
                tile = image[y0:y1, x0:x1]
                for key, val in self.lo_frag_paths.items():
                    frag = np.load(f"{val[0]}.npy")
                    mask = np.load(f"{val[1]}.npy")

                    frag_tile = frag[y0:y1, x0:x1]
                    if not tile_fully_in_fragment(mask, y0, y1, x0, x1):
                        continue
                    mse = (np.square(tile - frag_tile)).mean(axis=None)
                    print(mse)
                    if self.debug:
                        np.concatenate((tile, frag_tile, tile - frag_tile), axis=0)
                        cv.imwrite(f"./plots/tile_{y0}_{x0}_{key}.jpg", frag_tile)


    def save_final_img(self, img):
        """
        Saves the final image to the specified directory, either in the 'jp2', 'j2k', 'tiff', or 'tif' format if specified,
        or defaults to PNG. Handles saving in JPEG 2000 and TIFF formats using appropriate methods and raises an error for
        unsupported formats.

        Parameters:
            img: numpy.ndarray
                The image to be saved.

        Raises:
            NotImplementedError: If the save format is not one of the supported formats ('tif', 'tiff', 'jp2', or 'j2k').
        """
        if hasattr(self.config, 'save_format') and  self.config.save_format in ['jp2', 'j2k']:
            cv.imwrite(osp.join(self.out_dir, "final_stitch.png"), img)
            save_name = f"final_stitch.{self.config.save_format}"
            self.save_in_jp2(osp.join(self.out_dir, "final_stitch.png"), osp.join(self.out_dir, save_name))

        elif hasattr(self.config, 'save_format') and  self.config.save_format in ['tiff', 'tif']:
            save_name = f"final_stitch.{self.config.save_format}"
            min_val = img.min()
            max_val = img.max()
            rgb_norm = (img - min_val) / (max_val - min_val + 1e-8)
            tifffile.imwrite(osp.join(self.out_dir, save_name), rgb_norm)
            #cv.imwrite(osp.join(self.out_dir, save_name), img)

        else:
            cv.imwrite(osp.join(self.out_dir, "final_stitch.png"), img)
            save_name = f"final_stitch.{self.config.save_format}"
            self.save_in_jp2(osp.join(self.out_dir, "final_stitch.png"), osp.join(self.out_dir, save_name))
            raise NotImplementedError('Supported extensions: tif, tiff, jp2, j2k. Imaged saved as jp2')


    def save_in_jp2(self, i_path, o_pth):
        """
        Saves an image to the JP2 format using the `opj_compress` tool with specified
        compression and encoding options. Logs the process and handles potential
        errors occurring during the command execution.

        Parameters:
        i_path: str
            Input path of the image to be converted.
        o_pth: str
            Output path where the converted JP2 image will be saved.

        Raises:
        subprocess.CalledProcessError
            If the `opj_compress` command fails.

        """
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
            self.logger.error(f"opj_compress failed with exit code {e.returncode}")
            raise

    def run_homog(self, resize=False):

        # Load or estimate homographies
        if self.config.homog.load:
            with open(self.config.homog.load, "rb") as f:
                homographies = pickle.load(f)
        else:
            # The img paths is sent to load the images in correct format for feature extraction and matching
            # homographies, _ , to_del = self.homog_estimator.match(self.ref_resized_path, self.frag_paths)
            homographies, _, to_del = self.homog_estimator.register(self.ref_resized_path, self.frag_paths)
            # self.frag_paths = [val for idx, val in enumerate(self.frag_paths) if idx not in to_del]

            if self.config.homog.save:
                norm_homog = {}
                for idx, H in enumerate(homographies):
                    nH = scale_homog(H, self.final_scale)
                    norm_homog[self.frag_paths[idx].split('/')[-1]] = nH
                os.makedirs(self.config.homog.save, exist_ok=True)
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                with open(f"cache/homogs/opt_hom_{timestamp}.pkl", "wb") as f:
                    pickle.dump(norm_homog, f)

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
        Runs the optical flow computation process for a given image fragment and reference image.

        This method checks if the optical flow data is already available to be loaded from a specified
        path. If the data does not exist or loading is not configured, it computes the optical flow
        between a reference image and a warped fragment. The computed flow data is optionally saved
        to a specified path.

        Arguments:
            ref_path (str): The file path of the reference image used for optical flow computation.
            warped_frag (numpy.ndarray): The warped fragment image for which the optical flow is computed.
            frag_name (str): A unique name identifying the image fragment.
            resize (bool): Whether resizing is applied during the optical flow process (default is False).

        Returns:
            tuple: The first element is always `None` (reserved for future implementations), and the
                second element is a numpy array representing the computed optical flow.
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
        Equalizes the illumination of a flow fragment using a reference image. Provides optional resizing
        of the reference image and configurable tiling for memory-efficient light optimization.

        Args:
            ref_path (str): Path to the reference image.
            flow_fragment: The input fragment of flow to be equalized.
            frag_mask: The mask covering the valid areas of the fragment.
            resize (bool): Determines if the reference image should be resized before equalization.

        Returns:
            Tuple: A tuple containing the light-adjusted fragment and an optional mask. If tiling is used,
            the mask is always None.
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
        """
        Rectifies the reference image to a specified resolution using a perspective transform.

        Performs a perspective transformation to align and warp the reference image based on
        provided corner coordinates and final resolution settings. The rectified image is
        then saved to a predefined path and the reference image path is updated.

        Raises
        ------
        None

        Parameters
        ----------
        self : Self
            The instance of the class, providing access to configuration and reference
            image settings.

        Returns
        -------
        None
        """
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
        """
        Calculates and sets the processing parameters based on the final target resolution and aspect ratio.

        This method derives the height and width to be used during processing while maintaining the
        aspect ratio of the final target resolution. It adjusts the dimensions to match the required
        aspect ratio either by width-first or height-first calculation. The final scaling factor
        and the process dimensions are then updated and stored.

        Raises:
            None
        """
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
        Resizes a reference image to the specified dimensions and saves it to a cache
        location.

        The function reads the reference image from the provided path, resizes it to
        the given width and height, and saves the resized image to a specific directory.
        The new path is then stored for further usage.

        Parameters:
            size (tuple[int, int]): The target dimensions (height, width) to which
                the reference image will be resized.
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

def get_presets(path):
    return os.listdir(path)


def compose_configs(args):
    """
    Composes and merges configuration files from different sources including
    input configuration, default configuration, and preset configurations.

    The function performs the following steps:
    1. Loads the input configuration file from the specified input path.
    2. Loads the default configuration file.
    3. Loads a preset configuration file based on the preset name in the input
       configuration, or falls back to a default preset if none is specified.
    4. Merges the default and preset configurations.
    5. Merges the resulting configuration with the input configuration.
    6. Updates the configuration with specific paths for output and input
       directories.
    7. Creates the output directory if it does not exist.

    By combining configurations in the above manner, the function produces a
    final configuration object that can be used for further processing.

    Parameters:
        args (Namespace): A Namespace object containing the following attributes:
            - input (str): The path to the input directory.
            - output (str): The path to the output directory.

    Raises:
        FileNotFoundError: Raised if the input configuration file does not exist
            at the specified input path.

    Returns:
        DictConfig: The final composed and merged configuration object.
    """
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
        try:
            preset_cfg = OmegaConf.load(f"configs/presets/{input_cfg.preset_name}.yaml")
        except FileNotFoundError:
            presets = get_presets(f"configs/presets/")
            logger.error(f"Preset {input_cfg.preset_name} not found. Available presets: {presets}")
            raise FileNotFoundError(f"Preset {input_cfg.preset_name} not found. Available presets: {presets}")
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
    parser.add_argument("--presets", "-p", action="store_true", help="Prints available presets and exits")
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
    """
    This function serves as the main entry point for the application, handling initial setup,
    configuration processing, logging setup, and launching the core application functionality.

    Arguments:
        None

    Raises:
        OSError: If an error occurs while accessing or creating directories.
        FileNotFoundError: If a required file for configuration is missing.

    Returns:
        None
    """
    # Parse args
    args = args_process()
    # Check for presets and print them before running program

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





