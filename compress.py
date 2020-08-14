import numpy as np
import pandas as pd
import os, glob, time, datetime
import logging, pickle, argparse
import functools, itertools

from tqdm import tqdm, trange
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
    self.perceptual_loss = ps.PerceptualLoss(model='net-lin', net='alex', use_gpu=torch.cuda.is_available())

    # Load model
    device = helpers.get_device()
    logger = helpers.logger_setup(logpath=os.path.join(args.image_path, 'logs'), filepath=os.path.abspath(__file__))
    args, model, _ = helpers.load_model(args.ckpt_path, logger, device, model_mode=ModelModes.EVALUATION,
        current_args_d=None, prediction=True, strict=True)

    loader = datasets.EvalLoader(args.image_dir, normalize=args.normalize_input_image)
    n, N = 0, len(loader.dataset)
    input_filenames_total = list()
    output_filenames_total = list()
    bpp_total, q_bpp_total, LPIPS_total = torch.Tensor(N), torch.Tensor(N), torch.Tensor(N)

    with torch.no_grad():

        for idx, (data, bpp, filenames) in enumerate(tqdm(loader), 0):
            data = data.to(device, dtype=torch.float)
            B = data.size(0)
            reconstruction, q_bpp = model(data, writeout=False)
            perceptual_loss == self.perceptual_loss.forward(reconstruction, data, normalize=(not args.normalize_input_image))

            input_filenames_total.extend(filenames)

            for subidx in trange(reconstruction.shape[0]):
                fname = os.path.join(args.output_dir, "{}_RECON.png".format(filenames[subidx]))
                torchvision.utils.save_image(reconstruction[subidx], fname, normalize=True)
                output_filenames_total.append(fname)

            bpp_total[n:n + B] = bpp.data
            q_bpp_total[n:n + B] = q_bpp.data
            LPIPS_total[n:n + B] = perceptual_loss.data
            n += B

    df = pd.DataFrame([input_filenames_total, output_filenames_total]).T
    df.columns = ['input_filename', 'output_filename']
    df['bpp'] = bpp_total.cpu().numpy()
    df['q_bpp'] = q_bpp_total.cpu().numpy()
    df['LPIPS'] = LPIPS_total.cpu().numpy()

    df_path = os.path.join(output_dir, 'out.h5')
    df.to_hdf(df_path, key='df')

    logging.info('Complete. Reconstructions saved to {}. Output statistics saved to {}'.format(args.output_dir, df_path))


def main(**kwargs):

    description = "Compresses batch of images using specified learned model."
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-ckpt", "--ckpt_path", help="Path to model to be restored", type=str, required=True)
    parser.add_argument("-i", "--image_dir", help="Path to directory containing images to compress", type=str, 
        required=True)
    parser.add_argument("-o", "--output_dir", help="Path to directory to store output images", type=str,
        default='data/reconstructions')
    parser.add_argument('-bs', '--batch_size', help='dataloader batch size', type=int, default=8)
    args = parser.parse_args()

    # Launch training
    compress_batch(args)

if __name__ == '__main__':
    main()