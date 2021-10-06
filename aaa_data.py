import glob
import os
from itertools import chain
from multiprocessing import Lock
import natsort

import numpy as np
import cv2
from PIL import Image
import random
import torch
from torchvision.transforms import functional as F
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from skimage.util import random_noise

class aaaLoader(torch.utils.data.Dataset):
    """
    Implementation of torch.utils.data.Dataset backed by the HDF5 files, which iterates over the raw and label datasets
    patch by patch with a given stride.
    """

    def __init__(self,
                 file_path,
                 raw_path,
                 mask_path,
                 phase,
                 total_subject,
                 slice_train,
                 transformer_config
          ):

        self.file_path = file_path
        self.raw_path = raw_path
        self.mask_path = mask_path
        self.phase = phase
        self.total_subject = total_subject
        self.transformer = transformer_config
        self.slice_train = slice_train

        self.raw_all = []
        self.mask_all = []

        if self.phase!='test':
            for idx in self.total_subject:
                raw_path = os.path.join(self.raw_path,str(idx+1))
                mask_path = os.path.join(self.mask_path,str(idx+1))
                file_list = list(natsort.natsorted(os.listdir(raw_path)))

                file_raw = []
                file_mask = []

                for file_idx in file_list:
                    raw = Image.open(os.path.join(raw_path,str(file_idx)))
                    mask = Image.open(os.path.join(mask_path,str(file_idx)))
                    raw = np.array(raw)
                    mask = np.array(mask)

                    if len(file_raw)==0:
                        file_raw = raw
                        file_mask = mask

                    else:
                        file_raw = np.dstack((file_raw, raw))
                        file_mask = np.dstack((file_mask, mask))


                file_raw = np.transpose(file_raw,(2,0,1))
                file_mask = np.transpose(file_mask,(2,0,1))

                self.raw_all.append(file_raw)
                self.mask_all.append(file_mask)


        if self.phase == 'test':
            for idx in self.total_subject:
                raw_path = os.path.join(self.raw_path,str(idx+1))
                mask_path = os.path.join(self.mask_path,str(idx+1))
                file_list = list(natsort.natsorted(os.listdir(raw_path)))

                file_raw = []
                file_mask = []

                for file_idx in file_list:
                    raw = Image.open(os.path.join(raw_path,str(file_idx)))
                    mask = Image.open(os.path.join(mask_path,str(file_idx)))
                    raw = np.array(raw)
                    mask = np.array(mask)

                    if len(file_raw)==0:
                        file_raw = raw
                        file_mask = mask

                    else:
                        file_raw = np.dstack((file_raw, raw))
                        file_mask = np.dstack((file_mask, mask))


                file_raw = np.transpose(file_raw,(2,0,1))
                file_mask = np.transpose(file_mask,(2,0,1))

                self.raw_all.append(file_raw)
                self.mask_all.append(file_mask)


    def __getitem__(self, idx):
        prob = random.randint(1,11)
        execution_prob = 0.15
        # print("Subject index = %d"%idx)

        # all raw image & mask
        raw = self.raw_all[idx]
        mask = self.mask_all[idx]
        # mask = approximate_image(mask)

        raw_temp = np.zeros((self.slice_train, 256, 256)).astype(np.uint8)
        mask_temp = np.zeros((self.slice_train, 256, 256)).astype(np.uint8)

        depth = raw.shape[0]

        index_list = sorted(random.sample(range(0,depth), self.slice_train))

        if self.phase == 'train':

            for idx in range(self.slice_train):
                slice_index = index_list[idx]
                raw_slice = raw[slice_index]
                mask_slice = mask[slice_index]

                raw_temp[idx] = raw_slice
                mask_temp[idx] = mask_slice

            ##########################################################################################################

            raw_temp = raw_temp/255.
            raw_temp = raw_temp.astype(np.float32)

            mask_temp = approximate_image(mask_temp)
            mask_temp = mask_temp/255.
            mask_temp = mask_temp.astype(np.float32)

            # expand dimension
            raw = np.expand_dims(raw_temp, axis=0)
            mask = np.expand_dims(mask_temp, axis=0)

            return raw, mask

        # if self.phase == 'train':
        #     subject_depth = raw.shape[0]
        #     start_idx = random.randint(0, (subject_depth - self.slice_train))
        #
        #     # slicing in all image & mask
        #     # raw shape: depth * h * w
        #     raw = raw[start_idx:start_idx + self.slice_train]
        #     mask = mask[start_idx:start_idx + self.slice_train]
        #     mask = approximate_image(mask)
        #
        #     # data augmentation
        #     # Noise
        #     # Elastic deformation
        #     # crop 512 image size to 256 crop size
        #     # normalization (255 divided)
        #
        #     # sigma = random.randint(5, 12)
        #     # init_seed = None
        #     # for idx in range(raw.shape[0]):
        #     #     slice_index = index_list[idx]
        #     #     raw_slice = raw[slice_index]
        #     #     mask_slice = mask[slice_index]
        #     #
        #     #     if prob > 4:
        #     #         # [Data augmentation] Elastic_transform
        #     #         raw_slice, seed = add_elastic_transform(raw_slice, alpha=40, sigma=sigma, seed=init_seed, pad_size=30)
        #     #         if init_seed == None:
        #     #             init_seed = seed
        #     #         mask_slice, _ = add_elastic_transform(mask_slice, alpha=40, sigma=sigma, seed=seed, pad_size=30)
        #     #         mask_slice = approximate_image(mask_slice)
        #     #
        #     #     # [Data augmentation] Noise
        #     #
        #     #     if random.random() < execution_prob:
        #     #         raw_slice = random_noise(raw_slice, mode='gaussian', clip=True)
        #     #
        #     #     if random.random() < execution_prob:
        #     #         raw_slice = random_noise(raw_slice, mode='speckle', clip=True)
        #     #
        #     #     if random.random() < execution_prob:
        #     #         seed = random.uniform(0, 1)
        #     #         raw_slice = random_noise(raw_slice, mode='s&p', salt_vs_pepper=seed, clip=True)
        #     #
        #     #
        #     #     raw[idx] = raw_slice
        #     #     mask[idx] = mask_slice
        #
        #     ############################################################################################################
        #
        #     raw = raw / 255.
        #     raw = raw.astype(np.float32)
        #     mask = mask / 255.
        #     mask = mask.astype(np.float32)
        #
        #     # expand dimension
        #     raw = np.expand_dims(raw, axis=0)
        #     mask = np.expand_dims(mask, axis=0)
        #
        #     raw_transformed = raw
        #     label_transformed = mask
        #     return raw_transformed, label_transformed

        if self.phase == 'test':
            # print("idx = %d, depth = %d"%(idx, depth))
            # start slice number
            start_idx = 0
            if depth-start_idx > self.slice_train:
                # slicing in all image & mask
                raw = raw[start_idx:start_idx + self.slice_train]
                mask = mask[start_idx:start_idx + self.slice_train]

            mask = approximate_image(mask)
            raw = raw / 255
            raw = raw.astype(np.float32)
            mask = mask / 255
            mask = mask.astype(np.float32)

            # expand dimension
            raw = np.expand_dims(raw, axis=0)
            mask = np.expand_dims(mask, axis=0)

            raw_transformed = raw
            label_transformed = mask
            return raw_transformed, label_transformed


    @staticmethod
    def _transform_patches(datasets, label_idx, transformer):
        transformed_patches = []
        for dataset in datasets:
            # get the label data and apply the label transformer
            transformed_patch = transformer(dataset[label_idx])
            transformed_patches.append(transformed_patch)

        # if transformed_patches is a singleton list return the first element only
        if len(transformed_patches) == 1:
            return transformed_patches[0]
        else:
            return transformed_patches

    def __len__(self):
        return self.patch_count

    @staticmethod
    def _check_dimensionality(raws, labels):
        def _volume_shape(volume):
            if volume.ndim == 3:
                return volume.shape
            return volume.shape[1:]

        for raw, label in zip(raws, labels):
            assert raw.ndim in [3, 4], 'Raw dataset must be 3D (DxHxW) or 4D (CxDxHxW)'
            assert label.ndim in [3, 4], 'Label dataset must be 3D (DxHxW) or 4D (CxDxHxW)'

            assert _volume_shape(raw) == _volume_shape(label), 'Raw and labels have to be of the same size'

    @classmethod
    def create_datasets(cls, dataset_config, phase):
        phase_config = dataset_config[phase]

        file_paths = phase_config['file_paths']
        # load instance sampling configuration
        instance_ratio = phase_config.get('instance_ratio', None)
        random_seed = phase_config.get('random_seed', 0)

        datasets = []
        for file_path in file_paths:
            try:
                logger.info(f'Loading {phase} set from: {file_path}...')
                dataset = cls(file_path=file_path,
                              phase=phase,
                              mirror_padding=dataset_config.get('mirror_padding', None),
                              raw_internal_path=dataset_config.get('raw_path', 'raw'),
                              label_internal_path=dataset_config.get('label_path', 'label'),
                              weight_internal_path=dataset_config.get('weight_internal_path', None),
                              instance_ratio=instance_ratio, random_seed=random_seed)
                datasets.append(dataset)
            except Exception:
                logger.error(f'Skipping {phase} set: {file_path}', exc_info=True)
        return datasets

    @staticmethod
    def traverse_h5_paths(file_paths):
        assert isinstance(file_paths, list)
        results = []
        for file_path in file_paths:
            if os.path.isdir(file_path):
                # if file path is a directory take all H5 files in that directory
                iters = [glob.glob(os.path.join(file_path, ext)) for ext in ['*.h5', '*.hdf', '*.hdf5', '*.hd5']]
                for fp in chain(*iters):
                    results.append(fp)
            else:
                results.append(file_path)
        return results



def ds_stats(image):
    min_value, max_value, mean, std = calculate_stats(image)
    return min_value, max_value, mean, std

def calculate_stats(images):
    """
    Calculates min, max, mean, std given a list of ndarrays
    """
    # flatten first since the images might not be the same size
    flat = np.concatenate(
        [img.ravel() for img in images]
    )
    return np.min(flat), np.max(flat), np.mean(flat), np.std(flat)

def approximate_image(image):
    """
    Args:
        image : numpy array of image in datatype int16
    Return :
        image : numpy array of image in datatype uint8 only with 255 and 0
    """
    image[image > 127.5] = 255
    image[image < 127.5] = 0
    image = image.astype("uint8")
    return image


def add_elastic_transform(image, alpha, sigma, pad_size=30, seed=None):
    """
    Args:
        image : numpy array of image
        alpha : α is a scaling factor
        sigma :  σ is an elasticity coefficient
        random_state = random integer
        Return :
        image : elastically transformed numpy array of image
    """
    image = np.pad(image, pad_size, mode="symmetric")
    if seed is None:
        seed = random.randint(1, 100)
        random_state = np.random.RandomState(seed)
    else:
        random_state = np.random.RandomState(seed)
    shape = image.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1),
                         sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1),
                         sigma, mode="constant", cval=0) * alpha

    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))
    return cropping(map_coordinates(image, indices, order=1).reshape(shape), 256, pad_size, pad_size), seed

def cropping(image, crop_size, dim1, dim2):
   """crop the image and pad it to in_size
   Args :
       images : numpy array of images
       crop_size(int) : size of cropped image
       dim1(int) : vertical location of crop
       dim2(int) : horizontal location of crop
   Return :
       cropped_img: numpy array of cropped image
   """
   cropped_img = image[dim1:dim1 + crop_size, dim2:dim2 + crop_size]
   return cropped_img