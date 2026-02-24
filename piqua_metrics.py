import os
from distutils.command.config import config

os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = '10000000000'
import os.path as osp
import argparse
import shutil

import pyiqa as iqa

from concurrent.futures import ProcessPoolExecutor, as_completed
import subprocess

import pprint
import pickle

import cv2 as cv
import tifffile
import numpy as np
import torch.cuda
import datetime
from scipy.ndimage import map_coordinates
from scipy.optimize import least_squares
from omegaconf import OmegaConf

from modules.HomogEst import HomogEstimator
from modules.Stitcher import Stitcher, ActualBlender, DebugBlender
from modules.Optical import OpticalFlow
from modules.LightEqual import *
from utils.rectangularize import clip, order_points
from utils.utils import scale_homog

import gc

class Tester:
    def __init__(self):
        self.debug = True
        self.roi = {'minH': 7100, 'maxH': 8100, 'minW': 0, 'maxW': 500}
        self.brisque = iqa.create_metric("qalign_4bit", device='cuda')
        print(iqa.list_models())
        print(f"Is lower better: {self.brisque.lower_better}")

        os.makedirs('./plots/tiles', exist_ok=True)
        self.final_img_path = f'metrics/polokoule/final_stitch.png'
        self.frag_paths = os.listdir(f"metrics/polokoule/images")


    def run(self):

        image = np.asarray(cv.imread(self.final_img_path, cv.IMREAD_UNCHANGED))
        H, W = image.shape[:2]
        self.roi = {'minH': 0, 'maxH': H, 'minW': 0, 'maxW': W}
        th, tw = 2024, 2024
        tiles = [
            (y0, min(y0 + th, H), x0, min(x0 + tw, W))
            for y0 in range(self.roi['minH'], self.roi['maxH'], th)
            for x0 in range(self.roi['minW'], self.roi['maxW'], tw)
        ]

        best_met = 100000
        for eh, eH, ew, eW in tiles:
            tile = image[eh:eH, ew:eW, :] / 255
            tile_torch = torch.from_numpy(tile).permute((2,0,1)).unsqueeze(0).float()
            met_res = self.brisque(tile_torch).cpu().item()
            print(met_res)

            cv.imwrite(f'./plots/tiles/tile_{eh}_{ew}__{met_res}.jpg', image[eh:eH, ew:eW, :])


    def tile_fully_in_fragment(self, mask, y0, y1, x0, x1):
        """
        True if ALL pixels of the tile are inside the fragment mask.
        """
        if mask.ndim == 3:
            mask = mask[:, :, 0]

        tile = mask[y0:y1, x0:x1]


if __name__ == '__main__':
    tester = Tester()
    tester.run()