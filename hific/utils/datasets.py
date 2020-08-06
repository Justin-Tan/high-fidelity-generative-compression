"""
Authors:
@YannDubs 2019

Modifications:
Justin Tan 2019
"""


import subprocess
import os
import abc
import hashlib
import zipfile
import glob
import logging
import tarfile
import numpy as np

from skimage.io import imread
from scipy.stats import norm
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets

DIR = os.path.abspath(os.path.dirname(__file__))
COLOUR_BLACK = 0
COLOUR_WHITE = 1
DATASETS_DICT = {"mnist": "MNIST",
                 "fashion": "FashionMNIST",
                 "dsprites": "DSprites",
                 "dsprites_scream": "ScreamDSprites",
                 "celeba": "CelebA",
                 "chairs": "Chairs",
                 "custom": "Custom",
                 "jets": "Jets"}
DATASETS = list(DATASETS_DICT.keys())
SCREAM_PATH = '/home/jtan/gpu/jtan/github/disentangled/data/scream/scream.jpg'


def get_dataset(dataset):
    """Return the correct dataset."""
    dataset = dataset.lower()
    try:
        # eval because stores name as string in order to put it at top of file
        return eval(DATASETS_DICT[dataset])
    except KeyError:
        raise ValueError("Unknown dataset: {}".format(dataset))


def get_img_size(dataset):
    """Return the correct image size."""
    return get_dataset(dataset).img_size

def get_background(dataset):
    """Return the image background color."""
    return get_dataset(dataset).background_color


def get_dataloaders(dataset, train=True, root=None, shuffle=True, pin_memory=True, evaluate=False, 
                    batch_size=128, sampling_bias=False, logger=logging.getLogger(__name__), metrics=False, **kwargs):
    """A generic data loader

    Parameters
    ----------
    dataset : {"mnist", "fashion", "dsprites", "dsprites_scream", "celeba", "chairs"}
        Name of the dataset to load

    root : str
        Path to the dataset root. If `None` uses the default one.

    kwargs :
        Additional arguments to `DataLoader`. Default values are modified.
    """
    pin_memory = pin_memory and torch.cuda.is_available  # only pin if GPU available
    Dataset = get_dataset(dataset)

    if root is None:
        dataset = Dataset(logger=logger, train=train, sampling_bias=sampling_bias, metrics=metrics, evaluate=evaluate, **kwargs)
    else:
        dataset = Dataset(root=root, logger=logger, train=train, sampling_bias=sampling_bias, metrics=metrics, evaluate=evaluate, **kwargs)


    if sampling_bias is True:
        logger.info('Using biased sampler.')
        sampler = dataset.get_biased_sampler()
    else:
        sampler = None

    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      pin_memory=pin_memory,
                      sampler=sampler)  # **kwargs)

class Jets(torch.utils.data.TensorDataset):
    """Jets tabular dataset derived from LHC Olympics 2020

    Parameters
    ----------
    root : string
        Root directory of dataset.
    """

    files = {"train": "sig1k_fair/jets_all_train.h5", 
             "test": "sig1k_fair/jets_all_test.h5",
             "val": "sig1k_fair/jets_all_val.h5"}

    def __init__(self, root='/data/cephfs/punim0011/jtan/data/jets', logger=logging.getLogger(__name__), train=True,
        evaluate=False, pivots=['_Mjj'], auxillary=None, signal_region=False, sideband_region=False, context_dim=-1, **kwargs):

        self.root = root
        self.logger = logger
        self.train_data = os.path.join(root, type(self).files["train"])
        self.test_data = os.path.join(root, type(self).files["test"])
        self.val_data = os.path.join(root, type(self).files["val"])

        assert (train and evaluate) is False, 'Training and evaluation mode are incompatible!'

        if train is True:
            loadfile = self.train_data
        else:
            if evaluate is True:
                loadfile = self.test_data
            else:
                # Online validation
                loadfile = self.val_data

        # Note features scaled to (0,1) by default
        df, features, labels, pivots, pivot_names = _load_jets_data(loadfile, evaluate=evaluate, high_level=True,
            pivots=pivots, auxillary=auxillary, logger=logger, signal_region=signal_region, sideband_region=sideband_region)
        self.df = df


        if context_dim > 0:
            pivots = pivots[:,:context_dim]
            pivot_names = pivot_names[:context_dim]
        logger.info('Context Features: {}'.format(pivot_names))
        logger.info("# features: {}".format(features.shape[1]))
        logger.info("Specified context dim: {}".format(context_dim))

        N = features.shape[0]
        try:
            K = pivots.shape[1]
        except IndexError:
            K = 1

        self.input_dim = features.shape[-1]
        self.n_gen_factors = K
        self.pivot_names = pivot_names

        # Note labels is always the zeroth index, pivots start from 1 onward
        gen_factors = torch.cat([torch.Tensor(labels).view(N,1), torch.Tensor(pivots).view(N,K)], axis=-1) 
        logger.info("# gen factors (excluding labels): {}".format(K))
        self.tensors = torch.Tensor(features), gen_factors


class Custom(torch.utils.data.TensorDataset):
    """Custom tabular dataset specified under experimental section of paper

    Parameters
    ----------
    root : string
        Root directory of dataset.
    """

    files = {"train": "pivot_Mbc_train_small_scaled.h5", 
             "test": "pivot_Mbc_test_small_scaled.h5",
             "val": "pivot_Mbc_val_small_scaled.h5"}

    def __init__(self, root='/data/cephfs/punim0011/jtan/data', logger=logging.getLogger(__name__), train=True,
        evaluate=False, adversary=False, parquet=False, pivots=['_B_Mbc', '_B_deltaE'], auxillary=None, adv_n_classes=8, 
        signal_only=False, **kwargs):

        self.root = root
        self.logger = logger
        self.train_data = os.path.join(root, type(self).files["train"])
        self.test_data = os.path.join(root, type(self).files["test"])
        self.val_data = os.path.join(root, type(self).files["val"])

        assert (train and evaluate) is False, 'Training and evaluation mode are incompatible!'

        if train is True:
            loadfile = self.train_data
        else:
            if evaluate is True:
                loadfile = self.test_data
            else:
                # Online validation
                loadfile = self.val_data

        sidx_map = {k: v for k,v in zip(range(1, len(pivots)+1), pivots)}
        logger.info('Latent IDX mapping: {}'.format(sidx_map))
        if evaluate is True:
            df, features, labels, pivots = _load_custom_data(loadfile, evaluate=evaluate, adversary=adversary, parquet=parquet,
                pivots=pivots, auxillary=auxillary, adv_n_classes=adv_n_classes, logger=logger, signal_only=signal_only)
            self.df = df
        else:
            features, labels, pivots = _load_custom_data(loadfile, evaluate=evaluate, adversary=adversary, parquet=parquet,
                pivots=pivots, auxillary=auxillary, adv_n_classes=adv_n_classes, logger=logger, signal_only=signal_only)

        N = features.shape[0]
        try:
            K = pivots.shape[1]
        except IndexError:
            K = pivots.shape[0]
        logger.info("# features: {}".format(features.shape[1]))
        logger.info("# gen factors (excluding labels): {}".format(K))
        self.input_dim = features.shape[-1]
        self.n_gen_factors = K

        # Note labels is always the zeroth index, pivots start from 1 onward
        gen_factors = torch.cat([torch.Tensor(labels).view(N,1), torch.Tensor(pivots).view(N,K)], axis=-1) 
        self.tensors = torch.Tensor(features), gen_factors


class DisentangledDataset(Dataset, abc.ABC):
    """Base Class for disentangled VAE datasets.

    Parameters
    ----------
    root : string
        Root directory of dataset.

    transforms_list : list
        List of `torch.vision.transforms` to apply to the data when loading it.
    """

    def __init__(self, root, transforms_list=[], logger=logging.getLogger(__name__), **kwargs):
        self.root = root
        self.train_data = os.path.join(root, type(self).files["train"])
        self.transforms = transforms.Compose(transforms_list)
        self.logger = logger

        if not os.path.isdir(root):
            self.print("Downloading {} ...".format(str(type(self))))
            self.download()
            self.print("Finished Downloading.")

    def __len__(self):
        return len(self.imgs)

    def __ndim__(self):
        return self.imgs.size(1)

    @abc.abstractmethod
    def __getitem__(self, idx):
        """Get the image of `idx`.

        Return
        ------
        sample : torch.Tensor
            Tensor in [0.,1.] of shape `img_size`.
        """
        pass

    @abc.abstractmethod
    def download(self):
        """Download the dataset. """
        pass


class DSprites(DisentangledDataset):
    """DSprites Dataset from [1].

    Disentanglement test Sprites dataset.Procedurally generated 2D shapes, from 6
    disentangled latent factors. This dataset uses 6 latents, controlling the color,
    shape, scale, rotation and position of a sprite. All possible variations of
    the latents are present. Ordering along dimension 1 is fixed and can be mapped
    back to the exact latent values that generated that image. Pixel outputs are
    different. No noise added.

    The ground-truth factors of variation are (in the default setting):
        0 - shape (3 different values)
        1 - scale (6 different values)
        2 - orientation (40 different values)
        3 - position x (32 different values)
        4 - position y (32 different values)

    Notes
    -----
    - Link : https://github.com/deepmind/dsprites-dataset/
    - hard coded metadata because issue with python 3 loading of python 2

    Parameters
    ----------
    root : string
        Root directory of dataset.

    References
    ----------
    [1] Higgins, I., Matthey, L., Pal, A., Burgess, C., Glorot, X., Botvinick,
        M., ... & Lerchner, A. (2017). beta-vae: Learning basic visual concepts
        with a constrained variational framework. In International Conference
        on Learning Representations.

    """
    files = {"all": "dsprites.npz", "train": "dsprites_train.npz", "test": "dsprites_test.npz"}
    urls = {"train": "https://github.com/deepmind/dsprites-dataset/blob/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz?raw=true"}
    lat_names = ('shape', 'scale', 'orientation', 'posX', 'posY')
    lat_sizes = np.array([3, 6, 40, 32, 32])
    img_size = (1, 64, 64)
    n_gen_factors = len(lat_sizes)
    background_color = COLOUR_BLACK
    lat_values = {'posX': np.array([0., 0.03225806, 0.06451613, 0.09677419, 0.12903226,
                                    0.16129032, 0.19354839, 0.22580645, 0.25806452,
                                    0.29032258, 0.32258065, 0.35483871, 0.38709677,
                                    0.41935484, 0.4516129, 0.48387097, 0.51612903,
                                    0.5483871, 0.58064516, 0.61290323, 0.64516129,
                                    0.67741935, 0.70967742, 0.74193548, 0.77419355,
                                    0.80645161, 0.83870968, 0.87096774, 0.90322581,
                                    0.93548387, 0.96774194, 1.]),
                  'posY': np.array([0., 0.03225806, 0.06451613, 0.09677419, 0.12903226,
                                    0.16129032, 0.19354839, 0.22580645, 0.25806452,
                                    0.29032258, 0.32258065, 0.35483871, 0.38709677,
                                    0.41935484, 0.4516129, 0.48387097, 0.51612903,
                                    0.5483871, 0.58064516, 0.61290323, 0.64516129,
                                    0.67741935, 0.70967742, 0.74193548, 0.77419355,
                                    0.80645161, 0.83870968, 0.87096774, 0.90322581,
                                    0.93548387, 0.96774194, 1.]),
                  'scale': np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.]),
                  'orientation': np.array([0., 0.16110732, 0.32221463, 0.48332195,
                                           0.64442926, 0.80553658, 0.96664389, 1.12775121,
                                           1.28885852, 1.44996584, 1.61107316, 1.77218047,
                                           1.93328779, 2.0943951, 2.25550242, 2.41660973,
                                           2.57771705, 2.73882436, 2.89993168, 3.061039,
                                           3.22214631, 3.38325363, 3.54436094, 3.70546826,
                                           3.86657557, 4.02768289, 4.1887902, 4.34989752,
                                           4.51100484, 4.67211215, 4.83321947, 4.99432678,
                                           5.1554341, 5.31654141, 5.47764873, 5.63875604,
                                           5.79986336, 5.96097068, 6.12207799, 6.28318531]),
                  'shape': np.array([1., 2., 3.]),
                  'color': np.array([1.])}

    lat_values_max = {k: v.max() for k,v in lat_values.items()}
    factor_maxes = np.array([lat_values_max['shape'], lat_values_max['scale'], lat_values_max['orientation'], lat_values_max['posX'], lat_values_max['posY']])

    def __init__(self, root=os.path.join(DIR, '../data/dsprites/'), train=True, sampling_bias=False, **kwargs):
        super().__init__(root, [transforms.ToTensor()], **kwargs)


        self.sampling_bias = sampling_bias
        if self.sampling_bias is True:
            self.print('Using biased DSprites version.')
            # self.files = {"all": "dsprites_biased.npz", "train": "dsprites_train_biased.npz", "test": "dsprites_test_biased.npz"}
            self.files = {"all": "dsprites.npz", "train": "dsprites_train.npz", "test": "dsprites_test.npz"}

        self.save_path = os.path.join(self.root, self.files["all"])
        self.all_data = os.path.join(root, self.files["all"])
        self.train_data = os.path.join(root, self.files["train"])
        self.test_data = os.path.join(root, self.files["test"])
        # print('All: {}\nTrain: {}\nTest: {}'.format(self.all_data, self.train_data, self.test_data))

        if os.path.isfile(os.path.join(root, self.files["train"])) is False:
            self.print('Creating splits in folder {}'.format(self.root))
            self.create_splits(root=root)

        if kwargs['metrics'] is True:
            dataset_zip = np.load(self.all_data)
        else:
            if train is True:
                dataset_zip = np.load(self.train_data)
            else:
                dataset_zip = np.load(self.test_data)

        self.imgs = dataset_zip['imgs']
        self.lat_values = dataset_zip['latents_values']

    def download(self):
        """Download the dataset and create biased version."""
           
        os.makedirs(self.root)
        self.save_path = os.path.join(self.root, 'dsprites.npz')
        subprocess.check_call(["curl", "-L", type(self).urls["train"],
                               "--output", self.save_path])

        self.generate_biased_sample(self.root)

    
    def create_splits(self, root):
        from sklearn.model_selection import train_test_split

        ds = np.load(self.save_path)
        imgs_all = ds['imgs']
        values_all = ds['latents_values']
        classes_all = ds['latents_classes']

        imgs_train, imgs_test, values_train, values_test, classes_train, classes_test = train_test_split(imgs_all, values_all, classes_all, test_size=0.1)
        
        np.savez_compressed(self.all_data, imgs=imgs_all, latents_values=values_all, latents_classes=classes_all)
        np.savez_compressed(self.train_data, imgs=imgs_train, latents_values=values_train, latents_classes=classes_train)
        np.savez_compressed(self.test_data, imgs=imgs_test, latents_values=values_test, latents_classes=classes_test)

    def get_biased_sampler(self):

        from scipy.stats import multivariate_normal

        mu = np.array([0.5, 0.5])
        Sigma = np.eye(2)/25
        mvn = multivariate_normal(mu, Sigma)

        v = self.lat_values
        pos_xy = v[:,[4,5]]
        weights = np.ones(len(v))
        probs = mvn.pdf(pos_xy)

        # Only reweight images with class ellipse (now square)
        sq_idx = np.where(v[:,1]==1)[0]

        other_idx = np.where(v[:,1]!=1)[0]
        weights[sq_idx] = probs[sq_idx]
        # weights[other_idx] = np.clip(1 - probs[other_idx], 0.75, None)
        weights = torch.Tensor(weights)

        biased_sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights), replacement=True)

        return biased_sampler


    def generate_biased_sample(self, root):

        ds = np.load(self.save_path)

        imgs_all = ds['imgs']
        values_all = ds['latents_values']
        classes_all = ds['latents_classes']

        values_square = values_all[values_all[:,1]==1]
        imgs_square = imgs_all[values_all[:,1]==1]
        classes_square = classes_all[classes_all[:,1]==1]

        values_nonsquare = values_all[values_all[:,1]!=1]
        imgs_nonsquare = imgs_all[values_all[:,1]!=1]
        classes_nonsquare = classes_all[classes_all[:,1]!=1]

        values_square_x = values_square[:,4]

        probs = norm.pdf(values_square_x, loc=0.5, scale=0.2)
        w = probs/probs.sum()

        bidx = np.random.choice(values_square.shape[0], size=int(values_square_x.shape[0]/1.75), replace=False, p=w)
        biased_sample_values_square = values_square[bidx]
        biased_sample_imgs_square = imgs_square[bidx]
        biased_sample_classes_square = classes_square[bidx]

        imgs_all_biased = np.vstack([imgs_nonsquare, biased_sample_imgs_square])
        values_all_biased = np.vstack([values_nonsquare, biased_sample_values_square])
        classes_all_biased = np.vstack([classes_nonsquare, biased_sample_classes_square])

        np.savez_compressed(os.path.join(root, 'dsprites_biased.npz'), imgs=imgs_all_biased, latents_values=values_all_biased, latents_classes=classes_all_biased)


    def __getitem__(self, idx):
        """Get the image of `idx`
        Return
        ------
        sample : torch.Tensor
            Tensor in [0.,1.] of shape `img_size`.

        lat_value : np.array
            Array of length 6, that gives the value of each factor of variation.
        """
        # stored image have binary and shape (H x W) so multiply by 255 to get pixel
        # values + add dimension
        sample = np.expand_dims(self.imgs[idx] * 255, axis=-1)

        # ToTensor transforms numpy.ndarray (H x W x C) in the range
        # [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
        sample = self.transforms(sample)

        # Shape, Scale, Orientation, posX, posY
        lat_value = self.lat_values[idx][1:] / self.factor_maxes
        return sample, lat_value


class ScreamDSprites(DSprites):
    """Scream DSprites.
    This data set is the same as the original DSprites data set except that when
    sampling the observations X, a random patch of the Scream image is sampled as
    the background and the sprite is embedded into the image by inverting the
    color of the sampled patch at the pixels of the sprite.

    The ground-truth factors of variation are (in the default setting):
        0 - shape (3 different values)
        1 - scale (6 different values)
        2 - orientation (40 different values)
        3 - position x (32 different values)
        4 - position y (32 different values)
    """

    img_size = (3, 64, 64)

    def __init__(self, root=os.path.join(DIR, '../data/dsprites/'), **kwargs):
        super().__init__(root, **kwargs)
        with open(SCREAM_PATH, 'rb') as f:
            scream = Image.open(f)
            scream.thumbnail((350, 274, 3))
            self.scream = np.array(scream) * 1. / 255.
    
    def __getitem__(self, idx):
        """Get the image of `idx`
        Return
        ------
        sample : torch.Tensor
            (Color) Tensor in [0.,1.] of shape `img_size` - [N, C, H, W]

        lat_value : np.array
            Array of length 6, that gives the value of each factor of variation.
        """
        # stored image binary and shape (H x W)

        sample = self.imgs[idx]
        sample = np.repeat(sample[:, :, np.newaxis], 3, axis=2)

        # for i in range(sample.shape[0]):
        x_crop = np.random.randint(0, self.scream.shape[0] - 64)
        y_crop = np.random.randint(0, self.scream.shape[1] - 64)
        background = (self.scream[x_crop:x_crop + 64, y_crop:y_crop + 64] + 
                np.random.uniform(0, 1, size=3)) / 2.
        mask = (sample == 1)
        background[mask] = 1- background[mask]
        sample = background

        # Multiply by 255 to get pixel values
        # sample = sample * 255

        # ToTensor transforms numpy.ndarray (H x W x C) in the range
        # [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
        sample = self.transforms(sample)

        lat_value = self.lat_values[idx][1:] / self.factor_maxes
        return sample, lat_value
    



class CelebA(DisentangledDataset):
    """CelebA Dataset from [1].

    CelebFaces Attributes Dataset (CelebA) is a large-scale face attributes dataset
    with more than 200K celebrity images, each with 40 attribute annotations.
    The images in this dataset cover large pose variations and background clutter.
    CelebA has large diversities, large quantities, and rich annotations, including
    10,177 number of identities, and 202,599 number of face images.

    Notes
    -----
    - Link : http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

    Parameters
    ----------
    root : string
        Root directory of dataset.

    References
    ----------
    [1] Liu, Z., Luo, P., Wang, X., & Tang, X. (2015). Deep learning face
        attributes in the wild. In Proceedings of the IEEE international conference
        on computer vision (pp. 3730-3738).

    """
    urls = {"train": "https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/celeba.zip"}
    files = {"train": "img_align_celeba"}
    img_size = (3, 64, 64)
    background_color = COLOUR_WHITE

    def __init__(self, root=os.path.join(DIR, '../data/celeba'), **kwargs):
        super().__init__(root, [transforms.ToTensor()], **kwargs)

        self.imgs = glob.glob(self.train_data + '/*')

    def download(self):
        """Download the dataset."""
        save_path = os.path.join(self.root, 'celeba.zip')
        os.makedirs(self.root)
        subprocess.check_call(["curl", "-L", type(self).urls["train"],
                               "--output", save_path])

        hash_code = '00d2c5bc6d35e252742224ab0c1e8fcb'
        assert hashlib.md5(open(save_path, 'rb').read()).hexdigest() == hash_code, \
            '{} file is corrupted.  Remove the file and try again.'.format(save_path)

        with zipfile.ZipFile(save_path) as zf:
            self.print("Extracting CelebA ...")
            zf.extractall(self.root)

        os.remove(save_path)

        self.print("Resizing CelebA ...")
        preprocess(self.train_data, size=type(self).img_size[1:])

    def __getitem__(self, idx):
        """Get the image of `idx`

        Return
        ------
        sample : torch.Tensor
            Tensor in [0.,1.] of shape `img_size`.

        placeholder :
            Placeholder value as their are no targets.
        """
        img_path = self.imgs[idx]
        # img values already between 0 and 255
        img = imread(img_path)

        # put each pixel in [0.,1.] and reshape to (C x H x W)
        img = self.transforms(img)

        # no label so return 0 (note that can't return None because)
        # dataloaders requires so
        return img, 0


class Chairs(datasets.ImageFolder):
    """Chairs Dataset from [1].

    Notes
    -----
    - Link : https://www.di.ens.fr/willow/research/seeing3Dchairs

    Parameters
    ----------
    root : string
        Root directory of dataset.

    References
    ----------
    [1] Aubry, M., Maturana, D., Efros, A. A., Russell, B. C., & Sivic, J. (2014).
        Seeing 3d chairs: exemplar part-based 2d-3d alignment using a large dataset
        of cad models. In Proceedings of the IEEE conference on computer vision
        and pattern recognition (pp. 3762-3769).

    """
    urls = {"train": "https://www.di.ens.fr/willow/research/seeing3Dchairs/data/rendered_chairs.tar"}
    files = {"train": "chairs_64"}
    img_size = (1, 64, 64)
    background_color = COLOUR_WHITE

    def __init__(self, root=os.path.join(DIR, '../data/chairs'),
                 logger=logging.getLogger(__name__)):
        self.root = root
        self.train_data = os.path.join(root, type(self).files["train"])
        self.transforms = transforms.Compose([transforms.Grayscale(),
                                              transforms.ToTensor()])
        self.logger = logger

        if not os.path.isdir(root):
            self.print("Downloading {} ...".format(str(type(self))))
            self.download()
            self.print("Finished Downloading.")

        super().__init__(self.train_data, transform=self.transforms)

    def download(self):
        """Download the dataset."""
        save_path = os.path.join(self.root, 'chairs.tar')
        os.makedirs(self.root)
        subprocess.check_call(["curl", type(self).urls["train"],
                               "--output", save_path])

        self.print("Extracting Chairs ...")
        tar = tarfile.open(save_path)
        tar.extractall(self.root)
        tar.close()
        os.rename(os.path.join(self.root, 'rendered_chairs'), self.train_data)

        os.remove(save_path)

        self.print("Preprocessing Chairs ...")
        preprocess(os.path.join(self.train_data, '*/*'),  # root/*/*/*.png structure
                   size=type(self).img_size[1:],
                   center_crop=(400, 400))


class MNIST(datasets.MNIST):
    """Mnist wrapper. Docs: `datasets.MNIST.`"""
    img_size = (1, 32, 32)
    background_color = COLOUR_BLACK

    def __init__(self, train=True, root=os.path.join(DIR, '../data/mnist'), **kwargs):
        super().__init__(root,
                         train=train,
                         download=True,
                         transform=transforms.Compose([
                             transforms.Resize(32),
                             transforms.ToTensor()
                         ]))


class FashionMNIST(datasets.FashionMNIST):
    """Fashion Mnist wrapper. Docs: `datasets.FashionMNIST.`"""
    img_size = (1, 32, 32)

    def __init__(self, train=True, root=os.path.join(DIR, '../data/fashionMnist'), **kwargs):
        super().__init__(root,
                         train=train,
                         download=True,
                         transform=transforms.Compose([
                             transforms.Resize(32),
                             transforms.ToTensor()
                         ]))


# HELPERS
def preprocess(root, size=(64, 64), img_format='JPEG', center_crop=None):
    """Preprocess a folder of images.

    Parameters
    ----------
    root : string
        Root directory of all images.

    size : tuple of int
        Size (width, height) to rescale the images. If `None` don't rescale.

    img_format : string
        Format to save the image in. Possible formats:
        https://pillow.readthedocs.io/en/3.1.x/handbook/image-file-formats.html.

    center_crop : tuple of int
        Size (width, height) to center-crop the images. If `None` don't center-crop.
    """
    imgs = []
    for ext in [".png", ".jpg", ".jpeg"]:
        imgs += glob.glob(os.path.join(root, '*' + ext))

    for img_path in tqdm(imgs):
        img = Image.open(img_path)
        width, height = img.size

        if size is not None and width != size[1] or height != size[0]:
            img = img.resize(size, Image.ANTIALIAS)

        if center_crop is not None:
            new_width, new_height = center_crop
            left = (width - new_width) // 2
            top = (height - new_height) // 2
            right = (width + new_width) // 2
            bottom = (height + new_height) // 2

            img.crop((left, top, right, bottom))

        img.save(img_path, img_format)


def smooth_vars():
    return [
             'BB_gamma_hel',
             'B_CosTBTO',
             'B_CosTBz',
             'B_R2',
             'B_ThrustB',
             'B_ThrustO',
             'B_cms_cosTheta',
             'B_cms_daughterSumOf_pt',
             'B_gamma_cmsEnergyErr',
             'B_gamma_cms_E',
             'B_gamma_cms_clusterAbsZernikeMoment40',
             'B_gamma_cms_clusterAbsZernikeMoment51',
             'B_gamma_cms_clusterE1E9',
             'B_gamma_cms_clusterE9E21',
             'B_gamma_cms_clusterErrorE',
             'B_gamma_cms_clusterHighestE',
             'B_gamma_cms_clusterLAT',
             'B_gamma_cms_clusterPhi',
             'B_gamma_cms_clusterTheta',
             'B_gamma_cms_clusterUncorrE',
             'B_gamma_cms_cosTheta',
             'B_gamma_cms_eRecoil',
             'B_gamma_cms_m2Recoil',
             'B_gamma_cms_pRecoil',
             'B_gamma_cms_phi',
             'B_gamma_cms_pt',
             'B_useCMSFrame_bodaughterHighest_boE',
             'B_useCMSFrame_bodaughterHighest_bop',
             'B_useCMSFrame_bodaughterHighest_bopt',
             'B_useCMSFrame_bodaughterHighest_bopx',
             'B_useCMSFrame_bodaughterHighest_bopy',
             'B_useCMSFrame_bodaughterHighest_bopz',
             'B_useCMSFrame_bodaughterSumOf_bopt',
             'B_useCMSFrame_bodaughterSumOf_bopx',
             'B_useCMSFrame_bodaughterSumOf_bopy',
             'B_useCMSFrame_bodaughterSumOf_bopz']

def omit_vars():
    omit10 = ['B_useCMSFrame_bodaughterSumOf_bopx',
           'B_useCMSFrame_bodaughterSumOf_bopy',
           'B_useCMSFrame_bodaughterSumOf_bopz', 'B_gamma_cms_m2Recoil',
           'B_gamma_cms_E', 'B_CosTBTO',
           'B_useCMSFrame_bodaughterHighest_boE',
           'B_useCMSFrame_bodaughterHighest_bop', 'B_gamma_cms_pt',
           'B_useCMSFrame_bodaughterHighest_bopt']

    omit20 = ['B_ThrustB', 'B_cc1',
       'B_cms_cosAngleBetweenMomentumAndVertexVector',
       'B_cms_daughterSumOf_pt', 'B_useCMSFrame_bodaughterSumOf_bopt',
       'B_gamma_cms_clusterErrorE', 'B_gamma_cosMomVert',
       'B_gamma_cms_clusterUncorrE', 'B_gamma_cmsPyErr',
       'B_gamma_cms_eRecoil', 'B_gamma_cms_minC2TDist', 'B_cc2']
    
    omit30 = ['B_hso10', 'B_gamma_cms_clusterHighestE',
       'B_useCMSFrame_bodaughterHighest_bopx', 'B_gamma_cmsEnergyErr',
       'B_hso02', 'B_CosTBz', 'B_ThrustO', 'B_gamma_cmsPxErr',
       'B_useCMSFrame_bodaughterHighest_bopz']

    omit40 = ['B_hso02', 'B_useCMSFrame_bodaughterHighest_bopz', 'B_cc4',
       'B_useCMSFrame_bodaughterHighest_bopx', 'B_hso12',
       'B_gamma_cms_clusterR', 'B_gamma_cmsPxErr', 'B_gamma_cms_clusterTheta',
       'B_cms_ROE_eextra_cleanMask', 'B_gamma_cms_clusterE1E9']

    omit50 = ['B_gamma_cmsPzErr', 'B_et', 'B_cms_cosTheta', 'B_gamma_cmsEnergyErr',
              'B_gamma_cms_cosTheta', 'B_gamma_cms_clusterAbsZernikeMoment51',
                'BB_gamma_hel', 'B_ThrustO','B_gamma_cms_clusterZernikeMVA',
                'B_gamma_cms_clusterE9E21']

    omit45 =   ['BB_gamma_hel', 'B_ThrustO','B_gamma_cms_clusterZernikeMVA',
                'B_gamma_cms_clusterE9E21']
    omit55 = ['B_gamma_cmsPzErr', 'B_et', 'B_cms_cosTheta', 'B_gamma_cmsEnergyErr',
              'B_gamma_cms_cosTheta', 'B_gamma_cms_clusterAbsZernikeMoment51']
    omit60 = ['B_cms_phi', 'B_hso14', 'B_CosTBz', 'B_gamma_cms_clusterSecondMoment',
       'B_gamma_cms_clusterAbsZernikeMoment40', 'B_hso04',
       'B_gamma_cms_pRecoil', 'B_gamma_cms_clusterLAT', 'B_cc7', 'B_hso00']

    omit = omit10 + omit20 + omit30 + omit40 + omit55 #+ omit50
    return omit

def _load_custom_data(filename, evaluate=False, adversary=False, parquet=False,
    pivots=['_pivot'], auxillary=None, adv_n_classes=8, signal_only=False, logger=logging.getLogger(__name__)):

    """
    Loads tabular HEP dataset specified in experimental section of paper. 
    """
    from sklearn.preprocessing import MinMaxScaler
    minmaxscaler = MinMaxScaler()

    if parquet:
        import pyarrow.parquet as pq
        dataset = pq.ParquetDataset(filename)
        df = dataset.read(nthreads=4).to_pandas()
    else:
        df = pd.read_hdf(filename, key='df')

    if evaluate is False:
        df = df.sample(frac=1).reset_index(drop=True)

    if signal_only is True:
        print('Restricting dataset to signal events only.')
        n_original = df.shape[0]
        df = df[df._label > 0.5].reset_index(drop=True)
        print('Fraction of dataset: {:.3f}'.format(df.shape[0]/n_original))

    if auxillary is None:
        # Cleanup + omit variables prefixed with an underscore from training
        auxillary = [col for col in df.columns if col.startswith('_')]
        auxillary = list(set(auxillary))

    # CHANGE # OF FEATURES HERE
    K = 24
    df_features = df.drop(auxillary, axis=1)
    df_features = df_features[smooth_vars()[K:]]  # only smooth variables
    logger.info('Data shape: {}'.format(df_features.shape))
    logger.info('Features: {}'.format(df_features.columns.tolist()))

    pivots = ['_B_Mbc', '_B_deltaE']
    pivot_df = df[pivots]
    pivot_features = pivot_df[pivots]
    # Transform to range [0,1]
    scaled_pivot_features = minmaxscaler.fit_transform(pivot_features)

    if adversary:
        # Bin variable -> discrete classification problem
        # Each protected variable must be binned separately
        if len(pivots) == 1:
            pivot = pivots[0]
        pivot_df = pivot_df.assign(pivot_labels=pd.qcut(df[pivot], q=adv_n_classes, labels=False))
        pivot_labels = pivot_df.pivot_labels

    
    if evaluate is True:
        return df, np.nan_to_num(df_features.values), df._label.values.astype(np.int32), \
                np.squeeze(pivot_features.values.astype(np.float32))
    else:
        if adversary is True:
            return np.nan_to_num(df_features.values), df._label.values.astype(np.int32), \
                np.squeeze(pivot_features.values.astype(np.float32)), pivot_labels.values.astype(np.int32)
        else:
            # Return scaled pivots
            return np.nan_to_num(df_features.values), df._label.values.astype(np.int32), \
                np.squeeze(scaled_pivot_features.astype(np.float32))


def _load_jets_data(filename, evaluate=False, high_level=False, pivots=['_pivot'], auxillary=None, 
        EPS=0.01, signal_region=False, sideband_region=False, logger=logging.getLogger(__name__)):

    """
    Loads LHCOlympics 2020 Jets Dataset.
    TODO: Upload dataset to Zenodo
    """

    def _get_signal_region(df, SR_interval=(3.3,3.7)):
        # SR_interval defined in TeV, convert to GeV
        SR_check = lambda x: (x >= SR_interval[0]*1e3) and (x <= SR_interval[1]*1e3)
        in_SR = df._Mjj.apply(SR_check)
        df_SR, df_SB = df[in_SR], df[~in_SR]
        return df_SR, df_SB

    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    minmaxscaler = MinMaxScaler((EPS, 1.-EPS))

    df = pd.read_hdf(filename, key='df')

    if evaluate is False:
        df = df.sample(frac=1).reset_index(drop=True)

    if auxillary is None:
        # Cleanup + omit variables prefixed with an underscore from training
        auxillary = [col for col in df.columns if col.startswith('_')]
        auxillary = list(set(auxillary))

    # Get appropriate region, if requested
    if (signal_region or sideband_region) is True:
        assert (signal_region and sideband_region) is False, 'SR / SB options are mutually exclusive!'
        df_SR, df_SB = _get_signal_region(df)
        if signal_region is True:
            logger.info('Using events in SIGNAL REGION only')
            df = df_SR.reset_index(drop=True)
        else:
            logger.info('Using events in SIDEBAND REGION only')
            df = df_SB.reset_index(drop=True)

    # CHANGE # OF FEATURES HERE
    df_all_features = df.drop(auxillary, axis=1)  # large low-level feature set
    if 'label' in df_all_features.columns:
        df_all_features = df_all_features.drop(['label'], axis=1)

    if high_level is True:
        high_level_features = ['Mj1','Mj2']#,'deltaM','j1-tau21']  #['Mj2', 'deltaM', 'j1-tau21', 'j2-tau21']# + ['Mj1']  #, '_Mjj']
        df_features = df[high_level_features]
    else:
        df_features = df_all_features

    #df_features = df_features[df_features.columns[:12]]
    
    pivots = ['_Mjj']  #['deltaM', 'j1-tau21', 'j2-tau21', '_Mjj'] # ['_Mjj'] , 'Mj1', 'Mj2']
    # pivots = [col for col in df_all_features.columns if col not in high_level_features]
    pivot_df = df[pivots]
    pivot_features = pivot_df[pivots]

    logger.info('Data shape: {}'.format(df_features.shape))
    logger.info('Features: {}'.format(df_features.columns.tolist()))

    # Scale features to [EPS,1.-EPS] - account for transformation with inv det Jacobian
    from scipy.special import logit
    df_features = pd.DataFrame(logit(minmaxscaler.fit_transform(df_features)), columns=df_features.columns)
    pivot_features = pd.DataFrame(logit(minmaxscaler.fit_transform(pivot_features)), columns=pivot_features.columns)  # scale context
    #stdscaler = StandardScaler()
    #df_features = pd.DataFrame(logit(stdscaler.fit_transform(df_features)), columns=df_features.columns)

    return df, np.nan_to_num(df_features.values), df.label.values.astype(np.int32), \
        np.squeeze(pivot_features.values.astype(np.float32)), pivots

