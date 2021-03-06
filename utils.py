import collections
import importlib
import random
import cv2

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


def get_aaa_train_loader(config, train_sub, transform=None):
    loaders_config = config['aaa']

    batch_size = loaders_config.get('batch_size', 1)
    num_workers = loaders_config.get('num_workers', 1)

    all_npy_file = natsort.natsorted(os.listdir(os.path.join(loaders_config['prepro_path'], loaders_config['raw_path'])))
    all_npy_val_file = natsort.natsorted(os.listdir(os.path.join(loaders_config['prepro_path'], loaders_config['raw_path'])))

    val = train_sub[0]
    if config['trainer']['validation']:
        train_sub = [index for index in train_sub if index!=val]
    else:
        train_sub = [index for index in train_sub]

    # train_input = [index for index in all_npy_file if int(index.split(".")[0]) in train_sub]
    train_input = [index for index in all_npy_file if int(index.split("_")[0]) in train_sub]


    # train_val = [index for index in all_npy_val_file if int(index.split("_")[0])==val]

    npy_raw_path = os.path.join(loaders_config['prepro_path'], loaders_config['raw_path'])
    npy_mask_path = os.path.join(loaders_config['prepro_path'], loaders_config['mask_path'])
    print(npy_raw_path)
    print(npy_mask_path)

    npy_raw_val_path = os.path.join(loaders_config['prepro_path'], loaders_config['raw_test_path'])
    npy_mask_val_path = os.path.join(loaders_config['prepro_path'], loaders_config['mask_test_path'])

    dataset_train = aaa_data.aaaLoader(raw_path=npy_raw_path, mask_path=npy_mask_path, file_idx=train_input, mode=config['trainer']["mode"], transform=transform)
    # dataset_val = aaa_data.aaaLoader(raw_path=npy_raw_val_path, mask_path=npy_mask_val_path, file_idx=train_val, mode=config['trainer']["mode"])

    train_datasets = torch.utils.data.Subset(dataset_train, train_input)
    # val_datasets = torch.utils.data.Subset(dataset_val, train_val)

    train_datasets.dataset.transform = transform

    # when training with volumetric data use batch_size of 1 due to GPU memory constraints
    return {
        'train': DataLoader(train_datasets, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        # don't shuffle during validation: useful when showing how predictions for a given batch get better over time
        # 'val': DataLoader(val_datasets, batch_size=1, shuffle=False, num_workers=num_workers)
    }


def get_aaa_val_loader(config, test_sub):

    loaders_config = config['aaa']

    batch_size = loaders_config.get('batch_size', 1)
    num_workers = loaders_config.get('num_workers', 1)

    all_npy_file = natsort.natsorted(
        os.listdir(os.path.join(loaders_config['prepro_path'], loaders_config['raw_test_path'])))

    # test_list = [index for index in all_npy_file if int(index.split(".")[0]) in test_sub]
    test_list = [index for index in all_npy_file if int(index.split("_")[0]) in test_sub]
    npy_raw_path = os.path.join(loaders_config['prepro_path'], loaders_config['raw_test_path'])
    npy_mask_path = os.path.join(loaders_config['prepro_path'], loaders_config['mask_test_path'])

    dataset_test = aaa_data.aaaLoader(raw_path=npy_raw_path, mask_path=npy_mask_path, file_idx=test_list, mode="val")

    test_datasets = torch.utils.data.Subset(dataset_test, test_list)

    # when training with volumetric data use batch_size of 1 due to GPU memory constraints
    return {'val': DataLoader(test_datasets, batch_size=1, shuffle=False, num_workers=1)}


def get_aaa_test_loader(config, test_sub):

    loaders_config = config['aaa']

    batch_size = loaders_config.get('batch_size', 1)
    num_workers = loaders_config.get('num_workers', 1)

    all_npy_file = natsort.natsorted(
        os.listdir(os.path.join(loaders_config['prepro_path'], loaders_config['raw_test_path'])))

    test_list = [index for index in all_npy_file if int(index.split("_")[0]) in test_sub]
    npy_raw_path = os.path.join(loaders_config['prepro_path'], loaders_config['raw_test_path'])
    npy_mask_path = os.path.join(loaders_config['prepro_path'], loaders_config['mask_test_path'])

    dataset_test = aaa_data.aaaLoader(raw_path=npy_raw_path, mask_path=npy_mask_path, file_idx=test_list, mode=config['trainer']["mode"])

    test_datasets = torch.utils.data.Subset(dataset_test, test_list)

    # when training with volumetric data use batch_size of 1 due to GPU memory constraints
    return {'test': DataLoader(test_datasets, batch_size=1, shuffle=False, num_workers=1)}

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
        alpha : ?? is a scaling factor
        sigma :  ?? is an elasticity coefficient
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

    t_device = _move_to_device(t[:2])
    weight = None
    if len(t) == 2:
        input, target = t_device
        return input, target

    elif len(t) == 3:
        idx = t[2]
        input, target = t_device
        return input, target, idx



def eval_segmentation_visualization(config, pred, target, input_img, file_name="", sub_depth=[]):

    """
    Calculate only one person 3D volume evaluation function
    pred & target shape: Batch_size * 1 * Slice * H * W
    pred: 0 or 1 (np.uint32) after sigmoid function
    target: 0 or 1 (np.uint32)
    """

    input_img = np.array(input_img.data.cpu()) * 255
    pred[pred>0.5] = 1.
    pred[pred<=0.5] = 0.


    sub_id = int(file_name.split("_")[0])
    gt = np.load("/home/bh/AAA/3d_unet_AAA/data_1227/preprocess/mask_128/%s.npy"%sub_id)
    sub_sequential = int(file_name.split("_")[1].split(".")[0])
    sub_depth = gt.shape[0]

    quotient = sub_depth // config['aaa']['slice_num']
    remain = sub_depth % config['aaa']['slice_num']
    sub_sequential_check = np.arange(0,quotient)

    for batch in range (0, pred.shape[0]):
        batch_pred = pred[batch][0]
        batch_target = target[batch][0]
        batch_input = input_img[batch][0]
        if sub_sequential in sub_sequential_check:
            for slice in range(0,batch_pred.shape[0]):
                pred_slice = batch_pred[slice].astype(np.uint32)
                gt_slice = batch_target[slice].astype(np.uint32)
                input_slice = batch_input[slice].astype(np.uint8)

                ############################################################################################################
                ### Visualization

                pred_img = (pred_slice * 255).astype(np.uint8)
                gt_img = (gt_slice * 255).astype(np.uint8)

                img_pred_color = cv2.cvtColor(pred_img, cv2.COLOR_GRAY2BGR)
                img_gt_color = cv2.cvtColor(gt_img, cv2.COLOR_GRAY2BGR)
                img_raw_color = cv2.cvtColor(input_slice, cv2.COLOR_GRAY2BGR)


                green_pred = img_pred_color.copy()
                red_gt = img_gt_color.copy()

                idx_pred = np.where(green_pred > 0)
                idx_gt = np.where(red_gt > 0)
                red_gt[idx_gt[0], idx_gt[1], :] = [0, 0, 255]
                green_pred[idx_pred[0], idx_pred[1], :] = [0, 255, 0]

                img_overlap = img_gt_color.copy()
                img_overlap[:, :, 0] = 0
                img_overlap[:, :, 1] = pred_img

                add_img = cv2.addWeighted(img_raw_color, 0.7, img_overlap, 0.3, 0)

                cv2.putText(img_raw_color, "\"Raw image\"", (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                           cv2.LINE_AA, bottomLeftOrigin=False)
                cv2.putText(add_img, "\"Raw + GT + Predict\"", (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                           cv2.LINE_AA, bottomLeftOrigin=False)
                cv2.putText(red_gt, "\"GT\"", (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
                           bottomLeftOrigin=False)
                cv2.putText(green_pred, "\"Predict\"", (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                           cv2.LINE_AA, bottomLeftOrigin=False)
                cv2.putText(img_overlap, "\"GT + predict\"", (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                           cv2.LINE_AA, bottomLeftOrigin=False)

                img_all = np.concatenate([img_raw_color, add_img, red_gt, green_pred, img_overlap], axis=1)

                # Save image
                valid_path = config['aaa']['validation_path']
                valid_folder = "test_result"
                valid_mask_folder = "test_result_mask"
                if not os.path.exists(os.path.join(valid_path, valid_folder)):
                    os.mkdir(os.path.join(valid_path, valid_folder))
                    os.mkdir(os.path.join(valid_path, valid_mask_folder))

                valid_name = "%d_%d.png"%(sub_id, (sub_sequential*config['aaa']['slice_num']+slice))
                cv2.imwrite(os.path.join(valid_path,valid_folder,valid_name), img_all)
                cv2.imwrite(os.path.join(valid_path,valid_mask_folder,valid_name), pred_img)
                ############################################################################################################

        elif remain!=0 and sub_sequential not in sub_sequential_check:
            for slice in range(1,remain+1):
                pred_slice = batch_pred[-slice].astype(np.uint32)
                gt_slice = batch_target[-slice].astype(np.uint32)
                input_slice = batch_input[-slice].astype(np.uint8)

                ############################################################################################################
                ### Visualization

                pred_img = (pred_slice * 255).astype(np.uint8)
                gt_img = (gt_slice * 255).astype(np.uint8)

                img_pred_color = cv2.cvtColor(pred_img, cv2.COLOR_GRAY2BGR)
                img_gt_color = cv2.cvtColor(gt_img, cv2.COLOR_GRAY2BGR)
                img_raw_color = cv2.cvtColor(input_slice, cv2.COLOR_GRAY2BGR)

                green_pred = img_pred_color.copy()
                red_gt = img_gt_color.copy()

                idx_pred = np.where(green_pred > 0)
                idx_gt = np.where(red_gt > 0)
                red_gt[idx_gt[0], idx_gt[1], :] = [0, 0, 255]
                green_pred[idx_pred[0], idx_pred[1], :] = [0, 255, 0]

                img_overlap = img_gt_color.copy()
                img_overlap[:, :, 0] = 0
                img_overlap[:, :, 1] = pred_img

                add_img = cv2.addWeighted(img_raw_color, 0.7, img_overlap, 0.3, 0)

                cv2.putText(img_raw_color, "\"Raw image\"", (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                           cv2.LINE_AA, bottomLeftOrigin=False)
                cv2.putText(add_img, "\"Raw + GT + Predict\"", (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                           cv2.LINE_AA, bottomLeftOrigin=False)
                cv2.putText(red_gt, "\"GT\"", (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
                           bottomLeftOrigin=False)
                cv2.putText(green_pred, "\"Predict\"", (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                           cv2.LINE_AA, bottomLeftOrigin=False)
                cv2.putText(img_overlap, "\"GT + predict\"", (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                           cv2.LINE_AA, bottomLeftOrigin=False)

                img_all = np.concatenate([img_raw_color, add_img, red_gt, green_pred, img_overlap], axis=1)

                # Save image
                valid_path = config['aaa']['validation_path']
                valid_folder = "test_result"
                valid_mask_folder = "test_result_mask"
                if not os.path.exists(os.path.join(valid_path, valid_folder)):
                    os.mkdir(os.path.join(valid_path, valid_folder))
                    os.mkdir(os.path.join(valid_path, valid_mask_folder))

                valid_name = "%d_%d.png"%(sub_id, (sub_sequential*config['aaa']['slice_num']+slice-1))
                cv2.imwrite(os.path.join(valid_path,valid_folder,valid_name), img_all)
                cv2.imwrite(os.path.join(valid_path,valid_mask_folder,valid_name), pred_img)
                ############################################################################################################



def eval_segmentation_visualization_all(config, pred, target, input_img, file_name="", sub_depth=0):

    """
    Calculate only one person 3D volume evaluation function
    pred & target shape: Batch_size * 1 * Slice * H * W
    pred: 0 or 1 (np.uint32) after sigmoid function
    target: 0 or 1 (np.uint32)
    """

    input_img = np.array(input_img.data.cpu()) * 255
    pred[pred>0.5] = 1.
    pred[pred<=0.5] = 0.


    sub_id = int(file_name.split(".")[0])

    quotient = sub_depth // sub_depth
    remain = sub_depth % config['aaa']['slice_num']
    sub_sequential_check = np.arange(0,quotient)

    for batch in range (0, pred.shape[0]):
        batch_pred = pred[batch][0]
        batch_target = target[batch][0]
        batch_input = input_img[batch][0]

        for slice in range(0,batch_pred.shape[0]):
            pred_slice = batch_pred[slice].astype(np.uint32)
            gt_slice = batch_target[slice].astype(np.uint32)
            input_slice = batch_input[slice].astype(np.uint8)

            ############################################################################################################
            ### Visualization

            pred_img = (pred_slice * 255).astype(np.uint8)
            gt_img = (gt_slice * 255).astype(np.uint8)

            img_pred_color = cv2.cvtColor(pred_img, cv2.COLOR_GRAY2BGR)
            img_gt_color = cv2.cvtColor(gt_img, cv2.COLOR_GRAY2BGR)
            img_raw_color = cv2.cvtColor(input_slice, cv2.COLOR_GRAY2BGR)

            green_pred = img_pred_color.copy()
            red_gt = img_gt_color.copy()

            idx_pred = np.where(green_pred > 0)
            idx_gt = np.where(red_gt > 0)
            red_gt[idx_gt[0], idx_gt[1], :] = [0, 0, 255]
            green_pred[idx_pred[0], idx_pred[1], :] = [0, 255, 0]

            img_overlap = img_gt_color.copy()
            img_overlap[:, :, 0] = 0
            img_overlap[:, :, 1] = pred_img

            add_img = cv2.addWeighted(img_raw_color, 0.7, img_overlap, 0.3, 0)

            cv2.putText(img_raw_color, "\"Raw image\"", (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                       cv2.LINE_AA, bottomLeftOrigin=False)
            cv2.putText(add_img, "\"Raw + GT + Predict\"", (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                       cv2.LINE_AA, bottomLeftOrigin=False)
            cv2.putText(red_gt, "\"GT\"", (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
                       bottomLeftOrigin=False)
            cv2.putText(green_pred, "\"Predict\"", (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                       cv2.LINE_AA, bottomLeftOrigin=False)
            cv2.putText(img_overlap, "\"GT + predict\"", (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                       cv2.LINE_AA, bottomLeftOrigin=False)

            img_all = np.concatenate([img_raw_color, add_img, red_gt, green_pred, img_overlap], axis=1)

            # Save image
            valid_path = config['aaa']['validation_path']
            valid_folder = "test_result"
            valid_mask_folder = "test_result_mask"
            if not os.path.exists(os.path.join(valid_path, valid_folder)):
                os.mkdir(os.path.join(valid_path, valid_folder))
                os.mkdir(os.path.join(valid_path, valid_mask_folder))

            valid_name = "%d_%d.png"%(sub_id, slice)
            cv2.imwrite(os.path.join(valid_path,valid_folder,valid_name), img_all)
            cv2.imwrite(os.path.join(valid_path,valid_mask_folder,valid_name), pred_img)
            ############################################################################################################

