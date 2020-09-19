# Usage Guide

## Details

This repository defines a model for learnable image compression capable of compressing images of arbitrary size and resolution based on the paper ["High-Fidelity Generative Image Compression" (HIFIC) by Mentzer et. al.](https://hific.github.io/). There are three main components to this model, as described in the original paper:

1. An autoencoding architecture defining a nonlinear transform to latent space. This is used in place of the linear transforms used by traditional image codecs.
2. A hierarchical (two-level in this case) entropy model over the quantized latent representation enabling lossless compression through standard entropy coding.
3. A generator-discriminator component that encourages the decoder/generator component to yield realistic reconstructions.

The model is then trained end-to-end by optimization of a modified rate-distortion Lagrangian. Loosely, the model defines an 'amortization' of the storage requirements for an generic input image by learning the weights used in the neural compression/decompression scheme. The method is further described in the original paper [[0](https://arxiv.org/abs/2006.09965)]. The model is capable of yielding perceptually similar reconstructions to the input that tend to be more visually pleasing than standard image codecs which operate at comparable or higher bitrates.

The model weights range between 1.5-2GB on disk, making transmission of the model itself impractical. The idea is that the same model is instantiated and made available to a sender and receiver. The sender encodes messages into the compressed format, which is transmitted via some channel to the receiver, who then decodes the compressed representation into a lossy reconstruction of the original data.

This repository also includes a partial port of the [Tensorflow Compression library](https://github.com/tensorflow/compression) for general tools for neural image compression.

## Training

* Download a large (> 100,000) dataset of diverse color images. We found that using 1-2 training divisions of the [OpenImages](https://storage.googleapis.com/openimages/web/index.html) dataset was able to produce satisfactory results on arbitrary images. [Fabian Mentzer's L3C Repo](https://github.com/fab-jul/L3C-PyTorch/) provides utility functions for downloading and preprocessing OpenImages (the trained models did not use this exact split). Add the dataset path under the `DatasetPaths` class in `default_config.py`. Check default config/command line arguments:

```bash
vim default_config.py
python3 train.py -h
```

* For best results, as described in the paper, train an initial base model using the rate-distortion loss only, together with the hyperprior model, e.g. to target low bitrates:

```bash
# Train initial autoencoding model
python3 train.py --model_type compression --regime low --n_steps 1e6
```

* Then use the checkpoint of the trained base model to 'warmstart' the GAN architecture. Training the generator and discriminator from scratch was found to result in unstable training, but YMMV.

```bash
# Train using full generator-discriminator loss
python3 train.py --model_type compression_gan --regime low --n_steps 1e6 --warmstart --ckpt path/to/base/checkpoint
```

* Training after the warmstart for 2e5 steps using a batch size of 16 was sufficient to get reasonable results at sub-0.2 `bpp` per validation image, on average, using the default config in the `low` regime. You can change regimes to `med` or `high` to tradeoff perceptual quality for increased bitrate.

* Perceptual distortion metrics and `bpp` tend to decrease with a pareto-like distribution over training, so model quality can probably be significantly improved by training for an extremely large number of steps.

* If you get out-of-memory errors, try, in decreasing order of priority:
  * Decreasing the batch size (default 16).
  * Decreasing the number of channels of the latent representation (`latent_channels`, default 220). You may be able to reduce this quite aggressively as the network is highly over-parameterized - many values of the latent representation are near-deterministic.
  * Decreasing the number of channels used in the hyperprior.
  * Reducing the number of residual blocks in the generator (`n_residual_blocks`, default 7, the original paper used 9).
  * Training on smaller crops (`crop_size`, default `256 x 256`).

These options can be accessed through `/default_config.py`. While a larger architecture does not hurt performance, as the network can allow certain representations to be deterministic at near-zero entropy rates, decreasing the architecture size will allow for faster encoding/decoding.

* Logs for each experiment, including image reconstructions, are automatically created and periodically saved under `experiments/` with the appropriate name/timestamp. Metrics can be visualized via `tensorboard`:

```bash
tensorboard --logdir experiments/my_experiment/tensorboard --port 2401
```

Some sample logs for a couple of models can be found below:

* [Low bitrate regime (warmstart)](https://tensorboard.dev/experiment/xJV4hjbxRFy3TzrdYl7MXA/).
* [Low bitrate regime (full GAN loss)](https://tensorboard.dev/experiment/ETa0JIeOS0ONNZuNkIdrQw/).
* [High bitrate regime (full GAN loss)](https://tensorboard.dev/experiment/hAf1NYrqSVieKoDOcNpoGw/).

## Compression

* `compress.py` will compress generic images under some specified entropy model. This performs a forward pass through the model to obtain the compressed representation, optionally coding the representation using a vectorized rANS entropy coder, which is then saved to disk in binary format. As the model architecture is fully convolutional, compression will work with images of arbitrary size/resolution (subject to memory constraints).

* For message transmission, separate entropy models over the latents and hyperlatents must be instantiated and shared between sender and receiver.
  * The sender computes the bottleneck tensor and calls the `compress()` method in `src/model.py` to obtain the compressed representation for transmission.
  * The receiver calls the `decompress()` method in `src/model.py` to obtain the quantized bottleneck tensor, which is then passed through the generator to obtain the reconstruction.

* The compression scheme in hierarchial in the sense that two 'levels' of information representing the latent and hyperlatent variables must be compressed and stored in the message, together with the shape of the encoded data.
  * In addition to the compressed data, some metadata about the shape of the latent and hyperlatent representations must be stored for correct decoding when saving the binary format '`.hfc`' to disk, as shown in [`src/compression/compression_utils.load_compressed_format`](../src/compression/compression_utils.py).

```bash
# Check arguments
python3 compress.py -h

# Only get reconstructions
python3 compress.py -i path/to/image/dir -ckpt path/to/trained/model --reconstruct

# Save compressed format to disk
python3 compress.py -i path/to/image/dir -ckpt path/to/trained/model --save
```

* Optionally, reconstructions can be generated by passing the `--reconstruct` flag. Decoding without executing the rANS coder takes around 2-3 seconds for ~megapixel images on GPU, but this can definitely be optimized. As evaluating the CDF under the hyperprior entropy model involves a series of matrix multiplications, decoding is significantly faster on GPU.

* Executing the rANS coder is slow and represents a performance bottleneck. Passing the `--vectorize` flag is much faster, but incurs a constant-bit overhead as the elements of the vectorized message must be initialized to some value, which introduces redundancy into the compressed message. The batch size needs to be quite large to make this overhead negligible. A fix is in the pipeline.

## Pretrained Models

* Pretrained models using the OpenImages dataset can be found below. The examples at the end of this readme were produced using the `HIFIC-med` model. Each model was trained for around `2e5` warmup steps and `2e5` steps with the full generative loss. Note the original paper trained for `1e6` steps in each mode, so you can probably get better performance by training from scratch yourself.

* To use a pretrained model, download the selected model (~2 GB) and point the `-ckpt` argument in the command above to the corresponding path. If you want to finetune this model, e.g. on some domain-specific dataset, use the following options for each respective model (you will probably need to adapt the learning rate and rate-penalty schedule yourself):

| Target bitrate (bpp) | Weights | Training Instructions |
| ----------- | -------------------------------- | ---------------------- |
| 0.14 | [`HIFIC-low`](https://drive.google.com/open?id=1hfFTkZbs_VOBmXQ-M4bYEPejrD76lAY9) | <pre lang=bash>`python3 train.py --model_type compression_gan --regime low --warmstart -ckpt path/to/trained/model -nrb 9 -norm`</pre> |
| 0.30 | [`HIFIC-med`](https://drive.google.com/open?id=1QNoX0AGKTBkthMJGPfQI0dT0_tnysYUb) | <pre lang=bash>`python3 train.py --model_type compression_gan --regime med --warmstart -ckpt path/to/trained/model --likelihood_type logistic`</pre> |
| 0.45 | [`HIFIC-high`](https://drive.google.com/open?id=1BFYpvhVIA_Ek2QsHBbKnaBE8wn1GhFyA) | <pre lang=bash>`python3 train.py --model_type compression_gan --regime high --warmstart -ckpt path/to/trained/model -nrb 9 -norm`</pre> |

## Extensibility

* Network architectures can be modified by changing the respective files under `src/network`.
* The entropy model for both latents and hyperlatents can be changed by modifying `src/network/hyperprior`. For reference, there is an implementation of a discrete-logistic latent mixture model instead of the default latent mean-scale Gaussian model.
* The exact compression algorithm used can be replaced with any entropy coder that makes use of indexed probability tables. The default is a vectorized rANS coder which encodes overflow values using a variable-length code, but this behaviour is costly.

## Notes

* The reported `bpp` is the theoretical bitrate required to losslessly store the quantized latent representation of an image. Comparing this (not the size of the reconstruction) against the original size of the image will give you an idea of the reduction in memory footprint.
* The total size of the model (using the original architecture) is around 737 MB. Forward pass time should scale sublinearly provided everything fits in memory. A complete forward pass using a batch of 10 `256 x 256` images takes around 45s on a 2.8 GHz Intel Core i7.
* You may get an OOM error when compressing images which are too large (`>~ 4000 x 4000` on a typical consumer GPU). It's possible to get around this by splitting the input into distinct crops whose forward pass will fit in memory. We're working on a fix to automatically support this.
* Compression of >~ megapixel images takes around 8 GB of RAM.

## Contributing

Feel free to submit any questions/corrections/suggestions/bugs as issues. Pull requests are welcome. Thanks to Grace for helping refactor my code.

## Colab Demo
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Justin-Tan/high-fidelity-generative-compression/blob/master/assets/HiFIC_torch_colab_demo.ipynb)

To take a look under the hood, you can play with the [demonstration of the model in Colab](https://colab.research.google.com/github/Justin-Tan/high-fidelity-generative-compression/blob/master/assets/HiFIC_torch_colab_demo.ipynb), and compress your own images.

### References

The following additional papers were useful to understand implementation details.

0. Fabian Mentzer, George Toderici, Michael Tschannen, Eirikur Agustsson. High-Fidelity Generative Image Compression. [arXiv:2006.09965 (2020)](https://arxiv.org/abs/2006.09965).
1. Johannes Ballé, David Minnen, Saurabh Singh, Sung Jin Hwang, Nick Johnston. Variational image compression with a scale hyperprior. [arXiv:1802.01436 (2018)](https://arxiv.org/abs/1802.01436).
2. David Minnen, Johannes Ballé, George Toderici. Joint Autoregressive and Hierarchical Priors for Learned Image Compression. [arXiv 1809.02736 (2018)](https://arxiv.org/abs/1809.02736).
3. Johannes Ballé, Valero Laparra, Eero P. Simoncelli. End-to-end optimization of nonlinear transform codes for perceptual quality. [arXiv 1607.05006 (2016)](https://arxiv.org/abs/1607.05006).
4. Fabian Mentzer, Eirikur Agustsson, Michael Tschannen, Radu Timofte, Luc Van Gool. Practical Full Resolution Learned Lossless Image Compression. [arXiv 1811.12817 (2018)](https://arxiv.org/abs/1811.12817).

## TODO (priority descending)

* Investigate bit overhead in vectorized rANS implementation.
* Include `torchac` support for entropy coding.
* Rewrite rANS implementation for speed.
