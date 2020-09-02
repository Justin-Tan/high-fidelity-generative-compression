# high-fidelity-generative-compression

Pytorch implementation of the paper ["High-Fidelity Generative Image Compression" by Mentzer et. al.](https://hific.github.io/)

## Warning

This is a preliminary version. There may be sharp edges.

## Details

This repository defines a model for learnable image compression capable of compressing images of arbitrary size and resolution based on the paper ["High-Fidelity Generative Image Compression" (HIFIC) by Mentzer et. al.](https://hific.github.io/). There are three main components to this model, as described in the original paper:

1. An autoencoding architecture defining a nonlinear transform to latent space. This is used in place of the linear transforms used by traditional image codecs.
2. A hierarchical (two-level in this case) entropy model over the quantized latent representation enabling lossless compression through standard entropy coding.
3. A generator-discriminator component that encourages the decoder/generator component to yield realistic reconstructions.

The model is then trained end-to-end by optimization of a modified rate-distortion Lagrangian. Loosely, the model can be thought of as 'amortizing' the storage requirements for an generic input image through training a learnable compression/decompression scheme. The method is further described in the original paper [[0](https://arxiv.org/abs/2006.09965)]. The model is capable of yielding perceptually similar reconstructions to the input that tend to be more visually pleasing than standard image codecs which operate at comparable or higher bitrates.

## Examples

```bash
Original, 8.05 bpp / 2747 kB
```

![guess](assets/originals/camp_original.png)

```bash
HIFIC, 0.188 bpp / 64.1 kB
```

![guess](assets/hific/camp_hific.png)

```bash
JPG, 0.264 bpp / 90.1 kB
```

![guess](assets/camp_jpg_compress.png)

The image shown is an out-of-sample instance from the CLIC-2020 dataset. The HIFIC image is obtained by reconstruction via the learned model (checkpoint below). The JPG image is obtained by the `imagemagick` command `mogrify -format jpg -quality 42 camp_original.png`. Despite using around 1.5x the bitrate, the JPG image exhibits visible compression artifacts which are absent from the HIFIC-generated image.

Note that the learned model was not adapted in any way for evaluation on this image. More sample outputs from this model can be found at the end of the readme. All images are losslessly compressed to PNG format for viewing.

## Note

The generator is trained to achieve realistic and not exact reconstruction. It may synthesize certain portions of a given image to remove artifacts associated with lossy compression. Therefore, in theory **images which are compressed and decoded may be arbitrarily different from the input**. This precludes usage for sensitive applications. An important caveat from the authors is reproduced here: 

> "_Therefore, we emphasize that our method is not suitable for sensitive image contents, such as, e.g., storing medical images, or important documents._" 

## Usage

* Install Pytorch nightly and dependencies from [https://pytorch.org/](https://pytorch.org/). Then install other requirements.

```bash
pip install -r requirements.txt
```

* Clone this repository, `cd` in.

```bash
git clone https://github.com/Justin-Tan/high-fidelity-generative-compression.git
cd high-fidelity-generative-compression
```

To check if your setup is working, run `python3 -m src.model` in root.

### Training

* Download a large (> 100,000) dataset of diverse color images. We found that using 1-2 training divisions of the [OpenImages](https://storage.googleapis.com/openimages/web/index.html) dataset was able to produce satisfactory results on arbitrary images. Add the dataset path under the `DatasetPaths` class in `default_config.py`. Check default config/command line arguments:

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

* If you get out-of-memory errors, try, in decreasing order of priority:
  * Decreasing the batch size (default 16).
  * Decreasing the number of channels of the latent representation (`latent_channels`, default 220). You may be able to reduce this quite aggressively as the network is highly over-parameterized - many values of the latent representation are near-deterministic.
  * Reducing the number of residual blocks in the generator (`n_residual_blocks`, default 7, the original paper used 9).
  * Training on smaller crops (`crop_size`, default `256 x 256`).

* Logs for each experiment are automatically created and periodically saved under `experiments/` with the appropriate name/timestamp. Metrics can be visualized via `tensorboard`:

```bash
tensorboard --logdir experiments/my_experiment/tensorboard --port 2401
```

Some sample logs for a couple of models can be found below:

* [Low bitrate regime (warmstart)](https://tensorboard.dev/experiment/xJV4hjbxRFy3TzrdYl7MXA/).
* [Low bitrate regime (full GAN loss)](https://tensorboard.dev/experiment/ETa0JIeOS0ONNZuNkIdrQw/).
* [High bitrate regime (full GAN loss)](https://tensorboard.dev/experiment/hAf1NYrqSVieKoDOcNpoGw/).

### Compression 

* `compress.py` will compress generic images using some specified model. This performs a forward pass through the model to obtain the compressed representation, using a vectorized ANS entropy coder. As the model architecture is fully convolutional, this will work with images of arbitrary size/resolution (subject to memory constraints). Optionally, reconstructions from the compressed format can be generated by passing the `--reconstruct` flag. Decoding takes around 2-3 seconds for ~megapixel images on GPU, but this can probably be heavily optimized further, as the entropy coding is performed in Python currently.

```bash
# Check arguments
python3 compress.py -h

python3 compress.py -i path/to/image/dir -ckpt path/to/trained/model --reconstruct
```

### Pretrained Models

* Pretrained models using the OpenImages dataset can be found below. The examples at the end of this readme were produced using the `HIFIC-med` model. Each model was trained for around `2e5` warmup steps and `2e5` steps with the full generative loss, with a target bitrate of `bpp={'low': 0.14, 'med': 0.3, 'high': 0.45}`. Note the original paper trained for `1e6` steps in each mode, so you can probably get better performance by training from scratch yourself. 

* To use a pretrained model, download the model (around 2 GB) and point the `-ckpt` argument in the command above to the corresponding path. If you want to finetune this model, e.g. on some domain-specific dataset, use the following options for each respective model (you will probably need to adapt the learning rate and rate-penalty schedule yourself):

* [`HIFIC-low`](https://drive.google.com/open?id=1hfFTkZbs_VOBmXQ-M4bYEPejrD76lAY9)

```bash
# Low regime
python3 train.py --model_type compression_gan --regime low --warmstart -ckpt path/to/trained/model -nrb 9 -norm
```

* [`HIFIC-med`](https://drive.google.com/open?id=1QNoX0AGKTBkthMJGPfQI0dT0_tnysYUb)

```bash
# Medium regime
python3 train.py --model_type compression_gan --regime med --warmstart -ckpt path/to/trained/model --likelihood_type logistic
```

* [`HIFIC-high`](https://drive.google.com/open?id=1BFYpvhVIA_Ek2QsHBbKnaBE8wn1GhFyA)

```bash
# High regime
python3 train.py --model_type compression_gan --regime high --warmstart -ckpt path/to/trained/model -nrb 9 -norm
```

### Extensibility

* Network architectures can be modified by changing the respective files under `src/network`.
* The entropy model for both latents and hyperlatents can be changed by modifying `src/network/hyperprior`. For reference, there is an implementation of a discrete-logistic latent mixture model instead of the default latent mean-scale Gaussian model.

### Notes

* The reported `bpp` is the theoretical bitrate required to losslessly store the quantized latent representation of an image. Comparing this (not the size of the reconstruction) against the original size of the image will give you an idea of the reduction in memory footprint.
* The total size of the model (using the original architecture) is around 737 MB. Forward pass time should scale sublinearly provided everything fits in memory. A complete forward pass using a batch of 10 `256 x 256` images takes around 45s on a 2.8 GHz Intel Core i7.
* You may get an OOM error when compressing images which are too large (`>~ 4000 x 4000` on a typical consumer GPU). It's possible to get around this by splitting the input into distinct crops whose forward pass will fit in memory. We're working on a fix to automatically support this.

### Contributing

All content in this repository is licensed under the Apache-2.0 license. Feel free to submit any corrections or suggestions as issues.

### Examples

The samples below are taken from the CLIC2020 dataset, external to the training set. The reconstructions are produced using the above pretrained model (target bitrate `0.3 bpp`). It's interesting to try to guess which image is the original (images are saved as PNG for viewing). You can expand the spoiler tags below each image to reveal the answer.

A | B
:-------------------------:|:-------------------------:
![guess](assets/hific/CLIC2020_5_RECON_0.160bpp.png) | ![guess](assets/originals/CLIC2020_5.png)

<details>

  <summary>Image 1</summary>
  
  ```python
  Original: B (11.6 bpp) | HIFIC: A (0.160 bpp). Ratio: 72.5.
  ```

</details>

A             |  B
:-------------------------:|:-------------------------:
![guess](assets/originals/CLIC2020_20.png) | ![guess](assets/hific/CLIC2020_20_RECON_0.330bpp.png)

<details>

  <summary>Image 2</summary>

  ```python
  Original: A (14.6 bpp) | HIFIC: B (0.330 bpp). Ratio: 44.2
  ```

</details>

A | B
:-------------------------:|:-------------------------:
![guess](assets/originals/CLIC2020_18.png) | ![guess](assets/hific/CLIC2020_18_RECON_0.209bpp.png)

<details>

  <summary>Image 3</summary>
  
  ```python
  Original: A (12.3 bpp) | HIFIC: B (0.209 bpp). Ratio: 58.9
  ```
  
</details>

A             |  B
:-------------------------:|:-------------------------:
![guess](assets/hific/CLIC2020_19_RECON_0.565bpp.png) | ![guess](assets/originals/CLIC2020_19.png)

<details>

  <summary>Image 4</summary>
  
  ```python
  Original: B (19.9 bpp) | HIFIC: A (0.565 bpp). Ratio: 35.2
  ```
  
</details>

The last two show interesting failure modes: small figures in the distance are almost entirely removed (top of the central rock in the penultimate image), and the model bitrate increases significantly when the image is dominated by high-frequency components.

### Authors

* Grace Han
* Justin Tan

### Acknowledgements

* The code under `hific/perceptual_similarity/` implementing the perceptual distortion loss is modified from the [Perceptual Similarity repository](https://github.com/richzhang/PerceptualSimilarity).

### References

The following additional papers were useful to understand implementation details.

0. Fabian Mentzer, George Toderici, Michael Tschannen, Eirikur Agustsson. High-Fidelity Generative Image Compression. [arXiv:2006.09965 (2020)](https://arxiv.org/abs/2006.09965).
1. Johannes Ballé, David Minnen, Saurabh Singh, Sung Jin Hwang, Nick Johnston. Variational image compression with a scale hyperprior. [arXiv:1802.01436 (2018)](https://arxiv.org/abs/1802.01436).
2. David Minnen, Johannes Ballé, George Toderici. Joint Autoregressive and Hierarchical Priors for Learned Image Compression. [arXiv 1809.02736 (2018)](https://arxiv.org/abs/1809.02736).
3. Johannes Ballé, Valero Laparra, Eero P. Simoncelli. End-to-end optimization of nonlinear transform codes for perceptual quality. [arXiv 1607.05006 (2016)](https://arxiv.org/abs/1607.05006).
4. Fabian Mentzer, Eirikur Agustsson, Michael Tschannen, Radu Timofte, Luc Van Gool. Practical Full Resolution Learned Lossless Image Compression. [arXiv 1811.12817 (2018)](https://arxiv.org/abs/1811.12817).

## Citation

This is not the official implementation. Please cite the [original paper](https://arxiv.org/abs/2006.09965) if you use their work.

```bash
@article{mentzer2020high,
  title={High-Fidelity Generative Image Compression},
  author={Mentzer, Fabian and Toderici, George and Tschannen, Michael and Agustsson, Eirikur},
  journal={arXiv preprint arXiv:2006.09965},
  year={2020}
}
```
