# high-fidelity-generative-compression

Pytorch implementation of the paper ["High-Fidelity Generative Image Compression" by Mentzer et. al.](https://hific.github.io/). This repo also provides general utilities for lossless compression that interface with Pytorch.

## About

This repository defines a model for learnable image compression based on the paper ["High-Fidelity Generative Image Compression" (HIFIC) by Mentzer et. al.](https://hific.github.io/). The model is capable of compressing images of arbitrary spatial dimension and resolution up to two orders of magnitude in size, while maintaining perceptually similar reconstructions. Outputs tend to be more visually pleasing than standard image codecs operating at higher bitrates.

This repository also includes a partial port of the [Tensorflow Compression library](https://github.com/tensorflow/compression) - which provides general tools for neural image compression in Pytorch.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Justin-Tan/high-fidelity-generative-compression/blob/master/assets/HiFIC_torch_colab_demo.ipynb)

You can play with a [demonstration of the model in Colab](https://colab.research.google.com/github/Justin-Tan/high-fidelity-generative-compression/blob/master/assets/HiFIC_torch_colab_demo.ipynb), where you can upload and compress your own images.

## Example

Original | HiFIC
:-------------------------:|:-------------------------:
![guess](assets/originals/CLIC2020_5.png) | ![guess](assets/hific/CLIC2020_5_RECON_0.160bpp.png)

```python
Original: (6.01 bpp - 2100 kB) | HiFIC: (0.160 bpp - 56 kB). Ratio: 37.5.
```

The image shown is an out-of-sample instance from the CLIC-2020 dataset. The HiFIC image is obtained by reconstruction via a learned model provided below.

Note that the learned model was not adapted in any way for evaluation on this image. More sample outputs from this model can be found at the end of the README and in [EXAMPLES.md](assets/EXAMPLES.md).

## Note

The generator is trained to achieve realistic and not exact reconstruction. It may synthesize certain portions of a given image to remove artifacts associated with lossy compression. Therefore, in theory **images which are compressed and decoded may be arbitrarily different from the input**. This precludes usage for sensitive applications. An important caveat from the authors is reproduced here: 

> "_Therefore, we emphasize that our method is not suitable for sensitive image contents, such as, e.g., storing medical images, or important documents._" 

## Usage

* Install Pytorch nightly and dependencies from [https://pytorch.org/](https://pytorch.org/). Then install other requirements:

```bash
pip install -r requirements.txt
```

* Clone this repository, `cd` in:

```bash
git clone https://github.com/Justin-Tan/high-fidelity-generative-compression.git
cd high-fidelity-generative-compression
```

To check if your setup is working, run `python3 -m src.model` in root. Usage instructions can be found in the [user's guide](src/README.md).

### Training

* Download a large (> 100,000) dataset of diverse color images. We found that using 1-2 training divisions of the [OpenImages](https://storage.googleapis.com/openimages/web/index.html) dataset was able to produce satisfactory results on arbitrary images. Add the dataset path under the `DatasetPaths` class in `default_config.py`.

* For best results, as described in the paper, train an initial base model using the rate-distortion loss only, together with the hyperprior model, e.g. to target low bitrates:

```bash
# Train initial autoencoding model
python3 train.py --model_type compression --regime low --n_steps 1e6
```

* Then use the checkpoint of the trained base model to 'warmstart' the GAN architecture. Please see the [user's guide](src/README.md) for more detailed instructions.

```bash
# Train using full generator-discriminator loss
python3 train.py --model_type compression_gan --regime low --n_steps 1e6 --warmstart --ckpt path/to/base/checkpoint
```

### Compression

* `compress.py` will compress generic images using some specified model. This performs a forward pass through the model to yield the quantized latent representation, which is losslessly compressed using a vectorized ANS entropy coder and saved to disk in binary format. As the model architecture is fully convolutional, this will work with images of arbitrary size/resolution (subject to memory constraints).

```bash
python3 compress.py -i path/to/image/dir -ckpt path/to/trained/model --reconstruct
```
The compressed format can be transmitted and decoded using the routines in `compress.py`. The [Colab demo](https://colab.research.google.com/github/Justin-Tan/high-fidelity-generative-compression/blob/master/assets/HiFIC_torch_colab_demo.ipynb) illustrates the decoding process.

### Pretrained Models

* Pretrained model weights using the OpenImages dataset can be found below (~2 GB). The examples at the end of this readme were produced using the `HIFIC-med` model. The same models are also hosted in the following Zenodo repository: https://zenodo.org/record/4026003.

| Target bitrate (bpp) | Weights | Training Instructions |
| ----------- | -------------------------------- | ---------------------- |
| 0.14 | [`HIFIC-low`](https://drive.google.com/open?id=1hfFTkZbs_VOBmXQ-M4bYEPejrD76lAY9) | <pre lang=bash>`python3 train.py --model_type compression_gan --regime low --warmstart -ckpt path/to/trained/model -nrb 9 -norm`</pre> |
| 0.30 | [`HIFIC-med`](https://drive.google.com/open?id=1QNoX0AGKTBkthMJGPfQI0dT0_tnysYUb) | <pre lang=bash>`python3 train.py --model_type compression_gan --regime med --warmstart -ckpt path/to/trained/model --likelihood_type logistic`</pre> |
| 0.45 | [`HIFIC-high`](https://drive.google.com/open?id=1BFYpvhVIA_Ek2QsHBbKnaBE8wn1GhFyA) | <pre lang=bash>`python3 train.py --model_type compression_gan --regime high --warmstart -ckpt path/to/trained/model -nrb 9 -norm`</pre> |

## Examples

The samples below are taken from the CLIC2020 dataset, external to the training set. The bitrate is reported in bits-per-pixel (`bpp`). The reconstructions are produced using the above `HIFIC-med` model (target bitrate `0.3 bpp`). It's interesting to try to guess which image is the original (images are saved as PNG for viewing - best viewed widescreen). You can expand the spoiler tags below each image to reveal the answer.

For more examples see [EXAMPLES.md](assets/EXAMPLES.md). For even more examples see [this shared folder](https://drive.google.com/drive/folders/1lH1pTmekC1jL-gPi1fhEDuyjhfe5x6WG) (images within generated using the `HIFIC-low` model).

A             |  B
:-------------------------:|:-------------------------:
![guess](assets/originals/CLIC2020_3.png) | ![guess](assets/hific/CLIC2020_3_RECON_0.269bpp.png)

<details>

  <summary>Image 1</summary>

  ```python
  Original: A (11.8 bpp) | HIFIC: B (0.269 bpp). Ratio: 43.8
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

The last two show interesting failure modes: small figures in the distance are almost entirely removed (top of the central rock in the penultimate image), and the required model bitrate increases significantly when the image is dominated by high-frequency components.

### Authors

* Grace Han
* Justin Tan

### Acknowledgements

* The compression routines under `src/compression/` are derived from the [Tensorflow Compression library](https://github.com/tensorflow/compression).
* The vectorized rANS implementation used for entropy coding is based on the [Craystack repository](https://github.com/j-towns/craystack).
* The code under `src/loss/perceptual_similarity/` implementing the perceptual distortion loss is based on the [Perceptual Similarity repository](https://github.com/richzhang/PerceptualSimilarity).

### Contributing

All content in this repository is licensed under the Apache-2.0 license. Please open an issue if you encounter unexpected behaviour, or have corrections/suggestions to contribute.

## Citation

This is a PyTorch port of the original implementation. Please cite the [original paper](https://arxiv.org/abs/2006.09965) if you use their work.

```bash
@article{mentzer2020high,
  title={High-Fidelity Generative Image Compression},
  author={Mentzer, Fabian and Toderici, George and Tschannen, Michael and Agustsson, Eirikur},
  journal={arXiv preprint arXiv:2006.09965},
  year={2020}
}
```
