# high-fidelity-generative-compression
Generative Image Compression, the remix. Pytorch implementation of the paper ["High-Fidelity Generative Image Compression" by Mentzer et. al.](https://hific.github.io/)

## Warning
This is a preliminary version. There may be sharp edges.

## Note
The generator is trained to achieve realistic and not exact reconstruction. Therefore, in theory **images which are compressed and decoded may be arbitrarily different from the input**. This precludes usage for sensitive applications. An important caveat from the authors is reproduced here: 

> "_Therefore, we emphasize that our method is not suitable for sensitive image contents, such as, e.g., storing medical images, or important documents._" 

## Usage
* Install Pytorch and dependencies from [https://pytorch.org/](https://pytorch.org/). Then install other requirements.
```
pip install -r requirements.txt
```
* Clone this repository, `cd` in and view command line options.
```
git clone https://github.com/Justin-Tan/high-fidelity-generative-compression.git
cd high-fidelity-generative-compression

python3 train.py -h
```

### Notes
* The reported `bpp` is the theoretical bitrate required to losslessly store the quantized latent representation of an image as determined by the learned probability model provided by the hyperprior using some entropy coding algorithm. We're working on an rANS entropy coder.
* The "size" of the compressed image as reported in `bpp` does not account for the size of the model required to decode the compressed format.
* The total size of the model is around 737 MB. Forward pass time should scale sublinearly provided everything fits in memory.

## Contributing / Todo
All content in this repository is licensed under the Apache-2.0 license. Feel free to submit any corrections or suggestions as issues.

## Acknowledgements
* The code under `hific/perceptual_similarity/` implementing the perceptual distortion loss is modified from the [Perceptual Similarity repository](https://github.com/richzhang/PerceptualSimilarity).
* Kookaburra image (`data/kookaburra.jpg`) by [u/Crispy_Chooken](https://old.reddit.com/r/australia/comments/i3ffpk/best_photo_of_a_kookaburra_ive_taken_yet/).

## References
The following additional papers were useful to understand implementation details.
1. Johannes Ballé, David Minnen, Saurabh Singh, Sung Jin Hwang, Nick Johnston. Variational image compression with a scale hyperprior. [arXiv:1802.01436 (2018)](https://arxiv.org/abs/1802.01436).
2. David Minnen, Johannes Ballé, George Toderici. Joint Autoregressive and Hierarchical Priors for Learned Image Compression. [arXiv 1809.02736 (2018)](https://arxiv.org/abs/1809.02736).
3. Johannes Ballé, Valero Laparra, Eero P. Simoncelli. End-to-end optimization of nonlinear transform codes for perceptual quality. [arXiv 1607.05006 (2016)](https://arxiv.org/abs/1607.05006)

## Citation
This is not the official implementation. Please cite the [original paper](https://arxiv.org/abs/2006.09965) if you use their work.
```
@article{mentzer2020high,
  title={High-Fidelity Generative Image Compression},
  author={Mentzer, Fabian and Toderici, George and Tschannen, Michael and Agustsson, Eirikur},
  journal={arXiv preprint arXiv:2006.09965},
  year={2020}
}
```