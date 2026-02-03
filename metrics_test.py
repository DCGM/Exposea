import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = '10000000000'
import os.path as osp
import argparse
import shutil

from concurrent.futures import ProcessPoolExecutor, as_completed
import subprocess

import pprint
import pickle

import cv2 as cv
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


def _process_tile_worker(args):
    """
    Pickleable multiprocessing worker.
    Loads image/frag/mask via memmap for low RAM + no pickling big arrays.
    """
    (y0, y1, x0, x1,
     image_path,
     frag_items,   # list of (key, frag_path, mask_path)
     debug,
     out_dir) = args

    image = np.load(image_path, mmap_mode="r")
    tile = image[y0:y1, x0:x1]

    best_mse = None
    best_key = None
    best_frag_tile = None

    for key, frag_path, mask_path in frag_items:
        frag = np.load(frag_path, mmap_mode="r")
        mask = np.load(mask_path, mmap_mode="r")

        m = mask[:, :, 0] if mask.ndim == 3 else mask
        if not np.all(m[y0:y1, x0:x1]):
            continue

        frag_tile = frag[y0:y1, x0:x1]
        mse = np.square(tile - frag_tile).mean()

        if best_mse is None or mse < best_mse:
            best_mse = mse
            best_key = key
            if debug:
                best_frag_tile = frag_tile

    if debug and best_mse is not None and best_mse > 10.0 and best_frag_tile is not None:
        diff = tile - best_frag_tile
        debug_concat = np.concatenate((tile, best_frag_tile, diff), axis=0)

        out = debug_concat
        if out.dtype != np.uint8:
            out = np.clip(out, 0, 255).astype(np.uint8)

        os.makedirs(out_dir, exist_ok=True)
        cv.imwrite(os.path.join(out_dir, f"tile_{y0}_{x0}_{best_key}.jpg"), out)

    return (y0, x0, best_mse, best_key)

class Tester:
    def __init__(self, config):
        self.debug = True
        self.config = config

    def run(self):


        os.makedirs('./plots/tiles', exist_ok=True)
        final_img =  np.load(f"./metrics/{self.config.exp_name}/final_img.npy")
        with open(f"./metrics/{self.config.exp_name}/lo_frag_paths.pkl", "rb") as f:
            self.lo_frag_paths = pickle.load(f)
        final_img_path = f"./metrics/{self.config.exp_name}/final_img.npy"
        results = self.run_parallel_tiles_futures(final_img_path, self.lo_frag_paths, 50, 50, 12)

    def tile_fully_in_fragment(self, mask, y0, y1, x0, x1):
        """
        True if ALL pixels of the tile are inside the fragment mask.
        """
        if mask.ndim == 3:
            mask = mask[:, :, 0]

        tile = mask[y0:y1, x0:x1]
        return np.all(tile)

    def run_parallel_tiles_futures(self, image_path, lo_frag_paths, th, tw, max_workers=None):
        image = np.load(image_path, mmap_mode="r")
        H, W = image.shape[:2]

        # Make pickleable list of fragment file paths
        frag_items = []
        for key, val in lo_frag_paths.items():
            frag_items.append((key, f"{val[0]}.npy", f"{val[1]}.npy"))

        tiles = [(y0, min(y0 + th, H), x0, min(x0 + tw, W))
                 for y0 in range(0, H, th)
                 for x0 in range(0, W, tw)]

        out_dir = "./plots/tiles"

        tasks = [
            (y0, y1, x0, x1, image_path, frag_items, self.debug, out_dir)
            for (y0, y1, x0, x1) in tiles
        ]

        results = []
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(_process_tile_worker, t) for t in tasks]
            for f in as_completed(futures):
                results.append(f.result())

        results.sort(key=lambda r: (r[0], r[1]))
        for (y0, x0, best_mse, best_key) in results:
            if best_mse is None:
                print(f"Tile {y0},{x0}: no candidates")
            else:
                print(f"Tile {y0},{x0}: MSE {best_mse} | best {best_key}")

        return results


    def cif_image_tiles(self, image, tile_size, cand_bits=None):
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
                mses = []

                debug_tiles = []
                for key, val in self.lo_frag_paths.items():
                    frag = np.load(f"{val[0]}.npy")
                    mask = np.load(f"{val[1]}.npy")

                    frag_tile = frag[y0:y1, x0:x1]
                    if not tile_fully_in_fragment(mask, y0, y1, x0, x1):
                        continue

                    mse = (np.square(tile - frag_tile)).mean(axis=None)
                    mses.append(mse)
                    debug_tiles = [tile, frag_tile]

                if self.debug and np.min(mses) > 0.0:
                    tile = debug_tiles[0]
                    frag_tile = debug_tiles[1]
                    debug_concat = np.concatenate((tile, frag_tile, tile - frag_tile), axis=0)
                    cv.imwrite(f"./plots/tiles/tile_{y0}_{x0}_{key}.jpg", debug_concat)
                print(f"MSE: {np.min(mses)} | from {len(mses)}")


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
    app = Tester(config)
    app.run()

if __name__ == "__main__":
    main()
