import os
import numpy as np
import natsort
import yaml

from PIL import Image

###################################################################

# About Pre_data_saving
# This function stores input data for deepAAA in .npy format
# Each dataset has a different number of slices.
# So, it is divided by a fixed number of slices.
# Fixed number = 8 because 8 is the smallest data

###################################################################

def pre_train_data_saving(config):
    dst_path = './data/preprocess/'
    aaa_config = config['aaa']
    mini_slice = aaa_config['slice_num']

    raw_path = os.path.join(aaa_config['file_path'], aaa_config['raw_path'])
    mask_path = os.path.join(aaa_config['file_path'], aaa_config['mask_path'])
    print(raw_path)

    subject_list = natsort.natsorted(os.listdir(raw_path))

    # iteration of 60 subjects
    for sub_idx in subject_list:
        print("Subject idx = %s"%sub_idx)
        raw_path_c = os.path.join(raw_path, sub_idx)
        mask_path_c = os.path.join(mask_path, sub_idx)
        file_list = list(natsort.natsorted(os.listdir(raw_path_c)))

        file_raw = []
        file_mask = []

        # iteration file_list in subject folder
        for file_idx in file_list:
            raw = Image.open(os.path.join(raw_path_c, str(file_idx)))
            mask = Image.open(os.path.join(mask_path_c, str(file_idx)))
            raw = np.array(raw)
            mask = np.array(mask)

            if len(file_raw) == 0:
                file_raw = raw
                file_mask = mask

            else:
                file_raw = np.dstack((file_raw, raw))
                file_mask = np.dstack((file_mask, mask))

        file_raw = np.transpose(file_raw, (2, 0, 1))
        file_mask = np.transpose(file_mask, (2, 0, 1))

        img_depth_3d = file_raw.shape[0]
        print("* img depth = %d"%img_depth_3d)

        # saving slice image of .npy format
        for idx in range(0,img_depth_3d-mini_slice+1):
            raw_slice = file_raw[idx: idx+mini_slice]
            mask_slice = file_mask[idx: idx+mini_slice]
            np.save(os.path.join(dst_path,"raw_256_50","%s_%d.npy"%(sub_idx, idx)), raw_slice)
            np.save(os.path.join(dst_path,"mask_256_50","%s_%d.npy"%(sub_idx, idx)), mask_slice)

        print("=====")

#################
# For testing
# Subject_num.npy
def pre_test_data_saving(config):
    dst_path = './data/preprocess/'
    aaa_config = config['aaa']
    mini_slice = aaa_config['slice_num']

    raw_path = os.path.join(aaa_config['file_path'], aaa_config['raw_path'])
    mask_path = os.path.join(aaa_config['file_path'], aaa_config['mask_path'])
    print(raw_path)

    subject_list = natsort.natsorted(os.listdir(raw_path))

    # iteration of 60 subjects
    for sub_idx in subject_list:
        print("Subject idx = %s"%sub_idx)
        raw_path_c = os.path.join(raw_path, sub_idx)
        mask_path_c = os.path.join(mask_path, sub_idx)
        file_list = list(natsort.natsorted(os.listdir(raw_path_c)))

        file_raw = []
        file_mask = []

        # iteration file_list in subject folder
        for file_idx in file_list:
            raw = Image.open(os.path.join(raw_path_c, str(file_idx)))
            mask = Image.open(os.path.join(mask_path_c, str(file_idx)))
            raw = np.array(raw)
            mask = np.array(mask)

            if len(file_raw) == 0:
                file_raw = raw
                file_mask = mask

            else:
                file_raw = np.dstack((file_raw, raw))
                file_mask = np.dstack((file_mask, mask))

        file_raw = np.transpose(file_raw, (2, 0, 1))
        file_mask = np.transpose(file_mask, (2, 0, 1))

        img_depth_3d = file_raw.shape[0]
        print("* img depth = %d"%img_depth_3d)

        quotient = img_depth_3d // mini_slice
        remain = img_depth_3d % mini_slice
        final_idx = 0
        # saving slice image of .npy format
        for idx in range(0,quotient):
            raw_slice = file_raw[mini_slice*idx: mini_slice*idx+mini_slice]
            mask_slice = file_mask[mini_slice*idx: mini_slice*idx+mini_slice]
            np.save(os.path.join(dst_path,"raw_256_50_test","%s_%d.npy"%(sub_idx, idx)), raw_slice)
            np.save(os.path.join(dst_path,"mask_256_50_test","%s_%d.npy"%(sub_idx, idx)), mask_slice)
            final_idx = idx

        if remain!=0:
            final_idx = final_idx+1
            raw_slice = file_raw[-mini_slice:]
            mask_slice = file_mask[-mini_slice:]
            np.save(os.path.join(dst_path,"raw_256_50_test","%s_%d.npy"%(sub_idx, final_idx)), raw_slice)
            np.save(os.path.join(dst_path,"mask_256_50_test","%s_%d.npy"%(sub_idx, final_idx)), mask_slice)

        print("=====")


def load_config_yaml(config_file):
    return yaml.safe_load(open(config_file, 'r'))


if __name__ =="__main__":
    print("=== Training Start")
    config_file_path = "./config/train_config.yaml"
    config = load_config_yaml(config_file_path)
    pre_train_data_saving(config=config)
    pre_test_data_saving(config=config)