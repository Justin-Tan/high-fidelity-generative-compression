import numpy as np
import pandas as pd
import os, glob, time, datetime
import logging, pickle, argparse
import functools, itertools

from pprint import pprint
from tqdm import tqdm, trange
from collections import defaultdict

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

# Custom modules
from model import HificModel
from utils import helpers, datasets
import perceptual_similarity as ps
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
    perceptual_loss_fn = ps.PerceptualLoss(model='net-lin', net='alex', use_gpu=torch.cuda.is_available())

    # Load model
    device = helpers.get_device()
    logger = helpers.logger_setup(logpath=os.path.join(args.image_dir, 'logs'), filepath=os.path.abspath(__file__))
    loaded_args, model, _ = helpers.load_model(args.ckpt_path, logger, device, model_mode=ModelModes.EVALUATION,
        current_args_d=None, prediction=True, strict=False)

    dictify = lambda x: dict((n, getattr(x, n)) for n in dir(x) if not (n.startswith('__') or 'logger' in n))
    loaded_args_d, args_d = dictify(loaded_args), dictify(args)
    loaded_args_d.update(args_d)
    args = helpers.Struct(**loaded_args_d)
    logger.info(loaded_args_d)

    eval_loader = datasets.get_dataloaders('evaluation', root=args.image_dir, batch_size=args.batch_size,
                                           logger=logger, shuffle=False, normalize=args.normalize_input_image)

    n, N = 0, len(eval_loader.dataset)
    input_filenames_total = list()
    output_filenames_total = list()
    bpp_total, q_bpp_total, LPIPS_total = torch.Tensor(N), torch.Tensor(N), torch.Tensor(N)

    start_time = time.time()

    with torch.no_grad():

        for idx, (data, bpp, filenames) in enumerate(tqdm(eval_loader), 0):
            data = data.to(device, dtype=torch.float)
            B = data.size(0)
            reconstruction, q_bpp = model(data, writeout=False)
            perceptual_loss = perceptual_loss_fn.forward(reconstruction, data, normalize=(not args.normalize_input_image))

            input_filenames_total.extend(filenames)

            for subidx in range(reconstruction.shape[0]):
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

    df_path = os.path.join(args.output_dir, 'out.h5')
    df.to_hdf(df_path, key='df')

    pprint(df)

    logger.info('Complete. Reconstructions saved to {}. Output statistics saved to {}'.format(args.output_dir, df_path))
    delta_t = time.time() - start_time
    logger.info('Time elapsed: {:.3f} s'.format(delta_t))
    logger.info('Rate: {:.3f} Images / s:'.format(float(N) / delta_t))


def main(**kwargs):

    description = "Compresses batch of images using specified learned model."
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-ckpt", "--ckpt_path", type=str, required=True, help="Path to model to be restored")
    parser.add_argument("-i", "--image_dir", type=str, default='data/originals',
        help="Path to directory containing images to compress")
    parser.add_argument("-o", "--output_dir", type=str, default='data/reconstructions', 
        help="Path to directory to store output images")
    parser.add_argument('-bs', '--batch_size', type=int, default=1,
        help="Dataloader batch size. Set to 1 for images of different sizes.")
    args = parser.parse_args()

    input_images = glob.glob(os.path.join(args.image_dir, '*.jpg'))
    input_images += glob.glob(os.path.join(args.image_dir, '*.png'))

    assert len(input_images) > 0, 'No valid image files found in supplied directory!'

    print('Input images')
    pprint(input_images)
    # Launch training
    compress_batch(args)

if __name__ == '__main__':
    main()
