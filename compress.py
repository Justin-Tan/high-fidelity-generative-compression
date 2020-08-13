import numpy as np
import os, glob, time, datetime
import logging, pickle, argparse
import functools, itertools

from collections import defaultdict

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

# Custom modules
from hific.model import HificModel
from hific.utils import helpers, datasets
from default_config import hific_args, mse_lpips_args, directories, ModelModes, ModelTypes

def make_deterministic(seed=42):

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    np.random.seed(seed)

def compress_batch(args):

    # Reproducibility
    make_deterministic()

    # Load model
    device = helpers.get_device()
    logger = helpers.logger_setup(logpath=os.path.join(args.image_path, 'logs'), filepath=os.path.abspath(__file__))
    args, model, _ = helpers.load_model(args.ckpt_path, logger, device, current_args_d=None, prediction=True, strict=True)

    # Load image(s)
    loader = datasets.CustomLoader(args.image_dir)

def main(**kwargs):

    description = "Compresses batch of images using specified learned model."
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-ckpt", "--ckpt_path", help="Path to model to be restored", type=str)
    parser.add_argument("-i", "--image_dir", help="Path to directory containing images to compress", type=str)
    args = parser.parse_args()

    # Launch training
    compress_batch(args)

if __name__ == '__main__':
    main()