import os
import abc
import glob
import math
import logging
import numpy as np

from skimage.io import imread
import PIL
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets

DIR = os.path.abspath(os.path.dirname(__file__))
COLOUR_BLACK = 0
COLOUR_WHITE = 1
NUM_DATASET_WORKERS = 4
SCALE_MIN = 0.75
SCALE_MAX = 0.95
DATASETS_DICT = {"openimages": "OpenImages", "cityscapes": "CityScapes", 
                 "jetimages": "JetImages", "evaluation": "Evaluation"}
DATASETS = list(DATASETS_DICT.keys())

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

def exception_collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

def get_dataloaders(dataset, mode='train', root=None, shuffle=True, pin_memory=True, 
                    batch_size=8, logger=logging.getLogger(__name__), normalize=False, **kwargs):
    """A generic data loader

    Parameters
    ----------
    dataset : {"openimages", "jetimages", "evaluation"}
        Name of the dataset to load

    root : str
        Path to the dataset root. If `None` uses the default one.

    kwargs :
        Additional arguments to `DataLoader`. Default values are modified.
    """
    pin_memory = pin_memory and torch.cuda.is_available  # only pin if GPU available
    Dataset = get_dataset(dataset)

    if root is None:
        dataset = Dataset(logger=logger, mode=mode, normalize=normalize, **kwargs)
    else:
        dataset = Dataset(root=root, logger=logger, mode=mode, normalize=normalize, **kwargs)

    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      num_workers=NUM_DATASET_WORKERS,
                      collate_fn=exception_collate_fn,
                      pin_memory=pin_memory)


class BaseDataset(Dataset, abc.ABC):
    """Base Class for datasets.

    Parameters
    ----------
    root : string
        Root directory of dataset.

    transforms_list : list
        List of `torch.vision.transforms` to apply to the data when loading it.
    """

    def __init__(self, root, transforms_list=[], mode='train', logger=logging.getLogger(__name__),
         **kwargs):
        self.root = root
        
        try:
            self.train_data = os.path.join(root, self.files["train"])
            self.test_data = os.path.join(root, self.files["test"])
            self.val_data = os.path.join(root, self.files["val"])
        except AttributeError:
            pass

        self.transforms = transforms.Compose(transforms_list)
        self.logger = logger


        if not os.path.isdir(root):
            raise ValueError('Files not found in specified directory: {}'.format(root))

    def __len__(self):
        return len(self.imgs)

    def __ndim__(self):
        return tuple(self.imgs.size())

    @abc.abstractmethod
    def __getitem__(self, idx):
        """Get the image of `idx`.

        Return
        ------
        sample : torch.Tensor
            Tensor in [0.,1.] of shape `img_size`.
        """
        pass

class Evaluation(BaseDataset):
    """
    Parameters
    ----------
    root : string
        Root directory of dataset.

    """

    def __init__(self, root=os.path.join(DIR, 'data'), normalize=False, **kwargs):
        super().__init__(root, [transforms.ToTensor()], **kwargs)

        self.imgs = glob.glob(os.path.join(root, '*.jpg'))
        self.imgs += glob.glob(os.path.join(root, '*.png'))

        self.normalize = normalize

    def _transforms(self):
        """
        Up(down)scale and randomly crop to `crop_size` x `crop_size`
        """
        transforms_list = [transforms.ToTensor()]

        if self.normalize is True:
            transforms_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

        return transforms.Compose(transforms_list)

    def __getitem__(self, idx):
        """ TODO: This definitely needs to be optimized.
        Get the image of `idx`

        Return
        ------
        sample : torch.Tensor
            Tensor in [0.,1.] of shape `img_size`.

        """
        # img values already between 0 and 255
        img_path = self.imgs[idx]
        filename = os.path.splitext(os.path.basename(img_path))[0]
        filesize = os.path.getsize(img_path)
        try:
            img = PIL.Image.open(img_path)
            img = img.convert('RGB') 
            W, H = img.size  # slightly confusing
            bpp = filesize * 8. / (H * W)

            test_transform = self._transforms()
            transformed = test_transform(img)
        except:
            print('Error reading input images!')
            return None

        return transformed, bpp, filename

class OpenImages(BaseDataset):
    """OpenImages dataset from [1].

    Parameters
    ----------
    root : string
        Root directory of dataset.

    References
    ----------
    [1] https://storage.googleapis.com/openimages/web/factsfigures.html

    """
    files = {"train": "train", "test": "test", "val": "validation"}

    def __init__(self, root=os.path.join(DIR, 'data/openimages'), mode='train', crop_size=256, 
        normalize=False, **kwargs):
        super().__init__(root, [transforms.ToTensor()], **kwargs)

        if mode == 'train':
            data_dir = self.train_data
        elif mode == 'validation':
            data_dir = self.val_data
        else:
            raise ValueError('Unknown mode!')

        self.imgs = glob.glob(os.path.join(data_dir, '*.jpg'))
        self.imgs += glob.glob(os.path.join(data_dir, '*.png'))

        self.crop_size = crop_size
        self.image_dims = (3, self.crop_size, self.crop_size)
        self.scale_min = SCALE_MIN
        self.scale_max = SCALE_MAX
        self.normalize = normalize

    def _transforms(self, scale, H, W):
        """
        Up(down)scale and randomly crop to `crop_size` x `crop_size`
        """
        transforms_list = [# transforms.ToPILImage(),
                           transforms.RandomHorizontalFlip(),
                           transforms.Resize((math.ceil(scale * H), math.ceil(scale * W))),
                           transforms.RandomCrop(self.crop_size),
                           transforms.ToTensor()]

        if self.normalize is True:
            transforms_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

        return transforms.Compose(transforms_list)

    def __getitem__(self, idx):
        """ TODO: This definitely needs to be optimized.
        Get the image of `idx`

        Return
        ------
        sample : torch.Tensor
            Tensor in [0.,1.] of shape `img_size`.

        """
        # img values already between 0 and 255
        img_path = self.imgs[idx]
        filesize = os.path.getsize(img_path)
        try:
            # This is faster but less convenient
            # H X W X C `ndarray`
            # img = imread(img_path)
            # img_dims = img.shape
            # H, W = img_dims[0], img_dims[1]
            # PIL
            img = PIL.Image.open(img_path)
            img = img.convert('RGB') 
            W, H = img.size  # slightly confusing
            bpp = filesize * 8. / (H * W)

            shortest_side_length = min(H,W)

            minimum_scale_factor = float(self.crop_size) / float(shortest_side_length)
            scale_low = max(minimum_scale_factor, self.scale_min)
            scale_high = max(scale_low, self.scale_max)
            scale = np.random.uniform(scale_low, scale_high)

            dynamic_transform = self._transforms(scale, H, W)
            transformed = dynamic_transform(img)
        except:
            return None

        # apply random scaling + crop, put each pixel 
        # in [0.,1.] and reshape to (C x H x W)
        return transformed, bpp

class CityScapes(datasets.Cityscapes):
    """CityScapes wrapper. Docs: `datasets.Cityscapes.`"""
    img_size = (1, 32, 32)

    def _transforms(self, scale, H, W):
        """
        Up(down)scale and randomly crop to `crop_size` x `crop_size`
        """
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.Resize((math.ceil(scale * H), 
                               math.ceil(scale * W))),
            transforms.RandomCrop(self.crop_size),
            transforms.ToTensor(),
            ])

    def __init__(self, mode, root=os.path.join(DIR, 'data/cityscapes'), **kwargs):
        super().__init__(root,
                         split=mode,
                         transform=self._transforms(scale=np.random.uniform(0.5,1.0), 
                            H=512, W=1024))

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
        img = PIL.Image.open(img_path)
        width, height = img.size

        if size is not None and width != size[1] or height != size[0]:
            img = img.resize(size, PIL.Image.ANTIALIAS)

        if center_crop is not None:
            new_width, new_height = center_crop
            left = (width - new_width) // 2
            top = (height - new_height) // 2
            right = (width + new_width) // 2
            bottom = (height + new_height) // 2

            img.crop((left, top, right, bottom))

        img.save(img_path, img_format)
