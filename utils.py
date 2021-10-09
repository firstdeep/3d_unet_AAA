import collections
import importlib
import random


from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset, Dataset
from sklearn.model_selection import KFold


import aaa_data
import os
import natsort


class ConfigDataset(Dataset):
    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    @classmethod
    def create_datasets(cls, dataset_config, phase):
        """
        Factory method for creating a list of datasets based on the provided config.

        Args:
            dataset_config (dict): dataset configuration
            phase (str): one of ['train', 'val', 'test']

        Returns:
            list of `Dataset` instances
        """
        raise NotImplementedError

    @classmethod
    def prediction_collate(cls, batch):
        """Default collate_fn. Override in child class for non-standard datasets."""
        return default_prediction_collate(batch)


class SliceBuilder:
    """
    Builds the position of the patches in a given raw/label/weight ndarray based on the the patch and stride shape
    """

    def __init__(self, raw_datasets, label_datasets, weight_dataset, patch_shape, stride_shape, **kwargs):
        """
        :param raw_datasets: ndarray of raw data
        :param label_datasets: ndarray of ground truth labels
        :param weight_dataset: ndarray of weights for the labels
        :param patch_shape: the shape of the patch DxHxW
        :param stride_shape: the shape of the stride DxHxW
        :param kwargs: additional metadata
        """

        patch_shape = tuple(patch_shape)
        stride_shape = tuple(stride_shape)
        skip_shape_check = kwargs.get('skip_shape_check', False)
        if not skip_shape_check:
            self._check_patch_shape(patch_shape)

        self._raw_slices = self._build_slices(raw_datasets[0], patch_shape, stride_shape)
        if label_datasets is None:
            self._label_slices = None
        else:
            # take the first element in the label_datasets to build slices
            self._label_slices = self._build_slices(label_datasets[0], patch_shape, stride_shape)
            assert len(self._raw_slices) == len(self._label_slices)
        if weight_dataset is None:
            self._weight_slices = None
        else:
            self._weight_slices = self._build_slices(weight_dataset[0], patch_shape, stride_shape)
            assert len(self.raw_slices) == len(self._weight_slices)

    @property
    def raw_slices(self):
        return self._raw_slices

    @property
    def label_slices(self):
        return self._label_slices

    @property
    def weight_slices(self):
        return self._weight_slices

    @staticmethod
    def _build_slices(dataset, patch_shape, stride_shape):
        """Iterates over a given n-dim dataset patch-by-patch with a given stride
        and builds an array of slice positions.

        Returns:
            list of slices, i.e.
            [(slice, slice, slice, slice), ...] if len(shape) == 4
            [(slice, slice, slice), ...] if len(shape) == 3
        """
        slices = []
        if dataset.ndim == 4:
            in_channels, i_z, i_y, i_x = dataset.shape
        else:
            i_z, i_y, i_x = dataset.shape

        k_z, k_y, k_x = patch_shape
        s_z, s_y, s_x = stride_shape
        z_steps = SliceBuilder._gen_indices(i_z, k_z, s_z)
        for z in z_steps:
            y_steps = SliceBuilder._gen_indices(i_y, k_y, s_y)
            for y in y_steps:
                x_steps = SliceBuilder._gen_indices(i_x, k_x, s_x)
                for x in x_steps:
                    slice_idx = (
                        slice(z, z + k_z),
                        slice(y, y + k_y),
                        slice(x, x + k_x)
                    )
                    if dataset.ndim == 4:
                        slice_idx = (slice(0, in_channels),) + slice_idx
                    slices.append(slice_idx)
        return slices

    @staticmethod
    def _gen_indices(i, k, s):
        assert i >= k, 'Sample size has to be bigger than the patch size'
        for j in range(0, i - k + 1, s):
            yield j
        if j + k < i:
            yield i - k

    @staticmethod
    def _check_patch_shape(patch_shape):
        assert len(patch_shape) == 3, 'patch_shape must be a 3D tuple'
        assert patch_shape[1] >= 64 and patch_shape[2] >= 64, 'Height and Width must be greater or equal 64'


class FilterSliceBuilder(SliceBuilder):
    """
    Filter patches containing more than `1 - threshold` of ignore_index label
    """

    def __init__(self, raw_datasets, label_datasets, weight_datasets, patch_shape, stride_shape, ignore_index=(0,),
                 threshold=0.6, slack_acceptance=0.01, **kwargs):
        super().__init__(raw_datasets, label_datasets, weight_datasets, patch_shape, stride_shape, **kwargs)
        if label_datasets is None:
            return

        rand_state = np.random.RandomState(47)

        def ignore_predicate(raw_label_idx):
            label_idx = raw_label_idx[1]
            patch = np.copy(label_datasets[0][label_idx])
            for ii in ignore_index:
                patch[patch == ii] = 0
            non_ignore_counts = np.count_nonzero(patch != 0)
            non_ignore_counts = non_ignore_counts / patch.size
            return non_ignore_counts > threshold or rand_state.rand() < slack_acceptance

        zipped_slices = zip(self.raw_slices, self.label_slices)
        # ignore slices containing too much ignore_index
        filtered_slices = list(filter(ignore_predicate, zipped_slices))
        # unzip and save slices
        raw_slices, label_slices = zip(*filtered_slices)
        self._raw_slices = list(raw_slices)
        self._label_slices = list(label_slices)


class EmbeddingsSliceBuilder(FilterSliceBuilder):
    """
    Filter patches containing more than `1 - threshold` of ignore_index label and patches containing more than
    `patch_max_instances` labels
    """

    def __init__(self, raw_datasets, label_datasets, weight_datasets, patch_shape, stride_shape, ignore_index=(0,),
                 threshold=0.8, slack_acceptance=0.01, patch_max_instances=48, patch_min_instances=5, **kwargs):
        super().__init__(raw_datasets, label_datasets, weight_datasets, patch_shape, stride_shape, ignore_index,
                         threshold, slack_acceptance, **kwargs)

        if label_datasets is None:
            return

        rand_state = np.random.RandomState(47)

        def ignore_predicate(raw_label_idx):
            label_idx = raw_label_idx[1]
            patch = label_datasets[0][label_idx]
            num_instances = np.unique(patch).size

            # patch_max_instances is a hard constraint
            if num_instances <= patch_max_instances:
                # make sure that we have at least patch_min_instances in the batch and allow some slack
                return num_instances >= patch_min_instances or rand_state.rand() < slack_acceptance

            return False

        zipped_slices = zip(self.raw_slices, self.label_slices)
        # ignore slices containing too much ignore_index
        filtered_slices = list(filter(ignore_predicate, zipped_slices))
        # unzip and save slices
        raw_slices, label_slices = zip(*filtered_slices)
        self._raw_slices = list(raw_slices)
        self._label_slices = list(label_slices)


class RandomFilterSliceBuilder(EmbeddingsSliceBuilder):
    """
    Filter patches containing more than `1 - threshold` of ignore_index label and return only random sample of those.
    """

    def __init__(self, raw_datasets, label_datasets, weight_datasets, patch_shape, stride_shape, ignore_index=(0,),
                 threshold=0.8, slack_acceptance=0.01, patch_max_instances=48, patch_acceptance_probab=0.1,
                 max_num_patches=25, **kwargs):
        super().__init__(raw_datasets, label_datasets, weight_datasets, patch_shape, stride_shape,
                         ignore_index=ignore_index, threshold=threshold, slack_acceptance=slack_acceptance,
                         patch_max_instances=patch_max_instances, **kwargs)

        self.max_num_patches = max_num_patches

        if label_datasets is None:
            return

        rand_state = np.random.RandomState(47)

        def ignore_predicate(raw_label_idx):
            result = rand_state.rand() < patch_acceptance_probab
            if result:
                self.max_num_patches -= 1

            return result and self.max_num_patches > 0

        zipped_slices = zip(self.raw_slices, self.label_slices)
        # ignore slices containing too much ignore_index
        filtered_slices = list(filter(ignore_predicate, zipped_slices))
        # unzip and save slices
        raw_slices, label_slices = zip(*filtered_slices)
        self._raw_slices = list(raw_slices)
        self._label_slices = list(label_slices)


def get_class(class_name, modules):
    for module in modules:
        m = importlib.import_module(module)
        clazz = getattr(m, class_name, None)
        if clazz is not None:
            return clazz
    raise RuntimeError(f'Unsupported dataset class: {class_name}')


def _loader_classes(class_name):
    modules = [
        'pytorch3dunet.datasets.hdf5',
        'pytorch3dunet.datasets.dsb',
        'pytorch3dunet.datasets.utils'
    ]
    return get_class(class_name, modules)


def get_slice_builder(raws, labels, weight_maps, config):
    assert 'name' in config
    slice_builder_cls = _loader_classes(config['name'])
    return slice_builder_cls(raws, labels, weight_maps, **config)


def get_aaa_train_loader(config, train_sub):
    loaders_config = config['aaa']

    batch_size = loaders_config.get('batch_size', 1)
    batch_size_valid = int(batch_size/2)
    num_workers = loaders_config.get('num_workers', 1)

    all_npy_file = natsort.natsorted(os.listdir(os.path.join(loaders_config['prepro_path'], loaders_config['raw_path'])))

    train_input = [index for index in all_npy_file if int(index.split("_")[0]) in train_sub]
    total_num = len(train_input)
    valid_num = int(total_num * loaders_config['valid_ratio'])

    train_val = random.sample(train_input, valid_num)
    train_input = [index for index in train_input if index not in train_val]

    npy_raw_path = os.path.join(loaders_config['prepro_path'], loaders_config['raw_path'])
    npy_mask_path = os.path.join(loaders_config['prepro_path'], loaders_config['mask_path'])

    dataset_train = aaa_data.aaaLoader(raw_path=npy_raw_path, mask_path=npy_mask_path, file_idx=train_input)

    dataset_val = aaa_data.aaaLoader(raw_path=npy_raw_path, mask_path=npy_mask_path, file_idx=train_val)

    train_datasets = torch.utils.data.Subset(dataset_train, train_input)
    val_datasets = torch.utils.data.Subset(dataset_val, train_val)

    # when training with volumetric data use batch_size of 1 due to GPU memory constraints
    return {
        'train': DataLoader(train_datasets, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        # don't shuffle during validation: useful when showing how predictions for a given batch get better over time
        'val': DataLoader(val_datasets, batch_size=batch_size_valid, shuffle=False, num_workers=num_workers)
    }

def get_aaa_test_loader(config):
    assert 'loaders' in config, 'Could not find data loaders configuration'
    loaders_config = config['aaa']


    slice_num = int(loaders_config.get('slice_num',1))
    ratio = float(loaders_config.get('ratio_train_test',1))

    num_workers = loaders_config.get('num_workers', 1)

    batch_size = loaders_config.get('batch_size', 1)
    if torch.cuda.device_count() > 1 and not config['device'].type == 'cpu':
        batch_size = batch_size * torch.cuda.device_count()

    file_path = loaders_config.get("file_path", 1)
    raw_path = os.path.join(file_path,str(loaders_config.get("raw_path",1)))
    mask_path = os.path.join(file_path,str(loaders_config.get("mask_path",1)))

    total_subject = list(natsort.natsorted(os.listdir(raw_path)))
    total_subject = np.zeros((len(total_subject)))

    for i in range(0,total_subject.shape[0]):
        total_subject[i] = len(list(os.listdir(os.path.join(raw_path, str(i+1)))))

    subject_slice_idx = list(np.where(total_subject>=slice_num))[0]
    subject_len = len(subject_slice_idx)
    ratio = int(subject_len * ratio)

    subject_list = list(range(0,subject_len))
    train_idx = subject_list[ratio:]
    test_idx = [index for index in subject_list if index not in train_idx]

    transformer_config = loaders_config['test']['transformer']
    slice_train = loaders_config.get("slice_train",1)

    dataset = aaa_data.aaaLoader(file_path=file_path, raw_path=raw_path, mask_path=mask_path, phase='test', total_subject=subject_slice_idx,
                                                   slice_train=slice_train, transformer_config=transformer_config)
    print(test_idx)
    test_datasets = torch.utils.data.Subset(dataset, test_idx)

    # when training with volumetric data use batch_size of 1 due to GPU memory constraints
    return {'test': DataLoader(test_datasets, batch_size=batch_size, shuffle=False, num_workers=num_workers)}



def default_prediction_collate(batch):
    """
    Default collate_fn to form a mini-batch of Tensor(s) for HDF5 based datasets
    """
    error_msg = "batch must contain tensors or slice; found {}"
    if isinstance(batch[0], torch.Tensor):
        return torch.stack(batch, 0)
    elif isinstance(batch[0], tuple) and isinstance(batch[0][0], slice):
        return batch
    elif isinstance(batch[0], collections.Sequence):
        transposed = zip(*batch)
        return [default_prediction_collate(samples) for samples in transposed]

    raise TypeError((error_msg.format(type(batch[0]))))


def calculate_stats(images):
    """
    Calculates min, max, mean, std given a list of ndarrays
    """
    # flatten first since the images might not be the same size
    flat = np.concatenate(
        [img.ravel() for img in images]
    )
    return np.min(flat), np.max(flat), np.mean(flat), np.std(flat)


def sample_instances(label_img, instance_ratio, random_state, ignore_labels=(0,)):
    """
    Given the labelled volume `label_img`, this function takes a random subset of object instances specified by `instance_ratio`
    and zeros out the remaining labels.

    Args:
        label_img(nd.array): labelled image
        instance_ratio(float): a number from (0, 1]
        random_state: RNG state
        ignore_labels: labels to be ignored during sampling

    Returns:
         labelled volume of the same size as `label_img` with a random subset of object instances.
    """
    unique = np.unique(label_img)
    for il in ignore_labels:
        unique = np.setdiff1d(unique, il)

    # shuffle labels
    random_state.shuffle(unique)
    # pick instance_ratio objects
    num_objects = round(instance_ratio * len(unique))
    if num_objects == 0:
        # if there are no objects left, just return an empty patch
        return np.zeros_like(label_img)

    # sample the labels
    sampled_instances = unique[:num_objects]

    result = np.zeros_like(label_img)
    # keep only the sampled_instances
    for si in sampled_instances:
        result[label_img == si] = si

    return result

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


def count_parameter(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def split_training_batch(t, device):
    def _move_to_device(input):
        if isinstance(input, tuple) or isinstance(input, list):
            return tuple([_move_to_device(x) for x in input])
        else:
            return input.to(device)

    t = _move_to_device(t)
    weight = None
    if len(t) == 2:
        input, target = t
    else:
        input, target, weight = t
    return input, target, weight

def split_training_batch_validation(t, device):
    def _move_to_device(input):
        if isinstance(input, tuple) or isinstance(input, list):
            return tuple([_move_to_device(x) for x in input])
        else:
            return input.to(device)

    t = _move_to_device(t)
    weight = None
    if len(t) == 2:
        input, target = t

    return input, target

