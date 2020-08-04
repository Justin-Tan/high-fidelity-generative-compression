# high-fidelity-generative-compression
Generative Image Compression, the remix. Pytorch implementation of the paper ["High-Fidelity Generative Image Compression" by Mentzer et. al.](https://hific.github.io/)

# TODO
* Check input images format, [0,255], [0,1] or [-1,1]. LPIPS assumes [-1,1].

## Note
The generator is trained to achieve realistic and not exact reconstruction. Therefore, in theory **images which are compressed and decoded may be arbitrarily different from the input**. This precludes usage for sensitive applications. An important caveat from the authors is reproduced here: 

> "_Therefore, we emphasize that our method is not suitable for sensitive image contents, such as, e.g., storing medical images, or important documents._" 

## Usage
* Install PyTorch and dependencies from [https://pytorch.org/](https://pytorch.org/). Then install other requirements.
```
pip install -r requirements.txt
```
* Clone this repository, `cd` in and view command line options.
```
git clone https://github.com/Justin-Tan/high-fidelity-generative-compression.git
cd high-fidelity-generative-compression

python3 train.py -h
```
It should be noted that the "size" of the compressed image is this memory required to store the bitstring corresponding to the image and does not account for the storage of the model required to decode the compressed format.

## Contributing
All content in this repository is licensed under the Apache-2.0 license.

## Acknowledgements
* The code under `perceptual_similarity/` implementing the perceptual distortion loss is borrowed from the [Perceptual Similarity repository](https://github.com/richzhang/PerceptualSimilarity).

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