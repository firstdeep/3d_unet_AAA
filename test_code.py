import os
import natsort
import time
import numpy as np
import cv2

if __name__ =="__main__":

    # temp = np.random.randint(1,10,size=(5))
    # print(temp)
    # temp_minus = 1 - temp
    #
    # print(1-temp.mean())
    # print(temp_minus.mean())
    mask_path = '/home/bh/AAA/3d_unet_AAA/data_val/blood_77/'
    sub_list = natsort.natsorted(os.listdir(mask_path))

    pos = 0
    total = 0
    for sub_idx in sub_list:
        file_list = natsort.natsorted(os.listdir(os.path.join(mask_path, str(sub_idx))))
        file = []
        for idx in file_list:
            file.append(idx)
        print(len(file))
