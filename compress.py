import numpy as np
import pandas as pd
import os, glob, time, datetime
import logging, pickle, argparse
import functools, itertools

from pprint import pprint
from tqdm import tqdm, trange
from collections import defaultdict, namedtuple

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

# Custom modules
from src.helpers import utils, datasets
from src.compression import compression_utils
from src.loss.perceptual_similarity import perceptual_loss as ps
from default_config import hific_args, mse_lpips_args, directories, ModelModes, ModelTypes
from default_config import args as default_args

File = namedtuple('File', ['original_path', 'compressed_path',
                           'compressed_num_bytes', 'bpp'])

def make_deterministic(seed=42):

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False  # Don't go fast boi :(
    
    np.random.seed(seed)

def prepare_compress(model_path, input_dir, output_dir, batch_size=1):

    make_deterministic()

    input_images = glob.glob(os.path.join(input_dir, '*.jpg'))
    input_images += glob.glob(os.path.join(input_dir, '*.png'))
    assert len(input_images) > 0, 'No valid image files found in supplied directory!'
    print('Input images')
    pprint(input_images)

    device = utils.get_device()
    logger = utils.logger_setup(logpath=os.path.join(input_dir, 'logs'), filepath=os.path.abspath(__file__))
    loaded_args, model, _ = utils.load_model(args.ckpt_path, logger, device, model_mode=ModelModes.EVALUATION,
        current_args_d=None, prediction=True, strict=False)

    # Build probability tables
    model.Hyperprior.hyperprior_entropy_model.build_tables()

    # `batch_size` must be 1 for images of different shapes
    eval_loader = datasets.get_dataloaders('evaluation', root=input_dir, batch_size=batch_size,
                                           logger=logger, shuffle=False, normalize=loaded_args.normalize_input_image)
    utils.makedirs(output_dir)

    return model, loaded_args, eval_loader

def compress_and_save(model, args, data_loader, output_dir):
    # Compress and save compressed format to disk

    with torch.no_grad():
        for idx, (data, bpp, filenames) in enumerate(tqdm(eval_loader), 0):
            data = data.to(device, dtype=torch.float)
            assert data.size(0) == 1, 'Currently only supports saving single images.'

            # Perform entropy coding
            compressed_output = model.compress(data)

            out_path = os.path.join(output_dir, "{filenames[0]}_compressed.hfc")
            actual_bpp, theoretical_bpp = compression_utils.save_compressed_format(compressed_output,
                out_path=out_path)
            model.logger(f'Attained: {actual_bpp:.3f} bpp vs. theoretical: {theoretical_bpp:.3f} bpp.')


def load_and_decompress(model, compressed_format_path, out_path):
    # Decompress single image from compressed format on disk

    compressed_output = compression_utils.load_compressed_format(compressed_format_path)
    start_time = time.time()
    with torch.no_grad():
        reconstruction = model.decompress(compressed_output)

    q_bpp = compressed_output.total_bpp
    q_bpp = float(q_bpp.item()) if type(q_bpp) == torch.Tensor else float(q_bpp)

    torchvision.utils.save_image(reconstruction, out_path, normalize=True)
    delta_t = time.time() - start_time
    logger.info('Decoding time: {:.3f} s'.format(delta_t))
    model.logger(f'Reconstruction saved to {out_path}')

def compress_and_decompress(args):

    # Reproducibility
    make_deterministic()
    perceptual_loss_fn = ps.PerceptualLoss(model='net-lin', net='alex', use_gpu=torch.cuda.is_available())

    # Load model
    device = utils.get_device()
    logger = utils.logger_setup(logpath=os.path.join(args.image_dir, 'logs'), filepath=os.path.abspath(__file__))
    loaded_args, model, _ = utils.load_model(args.ckpt_path, logger, device, model_mode=ModelModes.EVALUATION,
        current_args_d=None, prediction=True, strict=False)

    # Override current arguments with recorded
    dictify = lambda x: dict((n, getattr(x, n)) for n in dir(x) if not (n.startswith('__') or 'logger' in n))
    loaded_args_d, args_d = dictify(loaded_args), dictify(args)
    loaded_args_d.update(args_d)
    args = utils.Struct(**loaded_args_d)
    logger.info(loaded_args_d)

    # Build probability tables
    model.Hyperprior.hyperprior_entropy_model.build_tables()


    eval_loader = datasets.get_dataloaders('evaluation', root=args.image_dir, batch_size=args.batch_size,
                                           logger=logger, shuffle=False, normalize=args.normalize_input_image)

    n, N = 0, len(eval_loader.dataset)
    input_filenames_total = list()
    output_filenames_total = list()
    bpp_total, q_bpp_total, LPIPS_total = torch.Tensor(N), torch.Tensor(N), torch.Tensor(N)
    utils.makedirs(args.output_dir)
    
    start_time = time.time()

    with torch.no_grad():

        for idx, (data, bpp, filenames) in enumerate(tqdm(eval_loader), 0):
            data = data.to(device, dtype=torch.float)
            B = data.size(0)
            input_filenames_total.extend(filenames)

            if args.reconstruct is True:
                # Reconstruction without compression
                reconstruction, q_bpp = model(data, writeout=False)
            else:
                # Perform entropy coding
                compressed_output = model.compress(data)

                if args.save is True:
                    assert B == 1, 'Currently only supports saving single images.'
                    compression_utils.save_compressed_format(compressed_output, 
                        out_path="{filenames[0]}_compressed.hfc")

                reconstruction = model.decompress(compressed_output)
                q_bpp = compressed_output.total_bpp

            if args.normalize_input_image is True:
                # [-1., 1.] -> [0., 1.]
                data = (data + 1.) / 2.

            perceptual_loss = perceptual_loss_fn.forward(reconstruction, data, normalize=True)


            for subidx in range(reconstruction.shape[0]):
                if B > 1:
                    q_bpp_per_im = float(q_bpp.cpu().numpy()[subidx])
                else:
                    q_bpp_per_im = float(q_bpp.item()) if type(q_bpp) == torch.Tensor else float(q_bpp)

                fname = os.path.join(args.output_dir, "{}_RECON_{:.3f}bpp.png".format(filenames[subidx], q_bpp_per_im))
                torchvision.utils.save_image(reconstruction[subidx], fname, normalize=True)
                output_filenames_total.append(fname)

            bpp_total[n:n + B] = bpp.data
            q_bpp_total[n:n + B] = q_bpp.data if type(q_bpp) == torch.Tensor else q_bpp
            LPIPS_total[n:n + B] = perceptual_loss.data
            n += B

    df = pd.DataFrame([input_filenames_total, output_filenames_total]).T
    df.columns = ['input_filename', 'output_filename']
    df['bpp_original'] = bpp_total.cpu().numpy()
    df['q_bpp'] = q_bpp_total.cpu().numpy()
    df['LPIPS'] = LPIPS_total.cpu().numpy()

    df_path = os.path.join(args.output_dir, 'compression_metrics.h5')
    df.to_hdf(df_path, key='df')

    pprint(df)

    logger.info('Complete. Reconstructions saved to {}. Output statistics saved to {}'.format(args.output_dir, df_path))
    delta_t = time.time() - start_time
    logger.info('Time elapsed: {:.3f} s'.format(delta_t))
    logger.info('Rate: {:.3f} Images / s:'.format(float(N) / delta_t))


def main(**kwargs):

    description = "Compresses batch of images using learned model specified via -ckpt argument."
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-ckpt", "--ckpt_path", type=str, required=True, help="Path to model to be restored")
    parser.add_argument("-i", "--image_dir", type=str, default='data/originals',
        help="Path to directory containing images to compress")
    parser.add_argument("-o", "--output_dir", type=str, default='data/reconstructions', 
        help="Path to directory to store output images")
    parser.add_argument('-bs', '--batch_size', type=int, default=1,
        help="Loader batch size. Set to 1 if images in directory are different sizes.")
    parser.add_argument("-rc", "--reconstruct", help="Reconstruct input image without compression.", action="store_true")
    parser.add_argument("-save", "--save", help="Save compressed format to disk.", action="store_true")
    args = parser.parse_args()

    input_images = glob.glob(os.path.join(args.image_dir, '*.jpg'))
    input_images += glob.glob(os.path.join(args.image_dir, '*.png'))

    assert len(input_images) > 0, 'No valid image files found in supplied directory!'

    print('Input images')
    pprint(input_images)

    compress_and_decompress(args)

if __name__ == '__main__':
    main()
