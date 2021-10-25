import os
import natsort
import time
import numpy as np
import cv2

if __name__ =="__main__":

    temp = np.random.randint(1,10,size=(5))
    print(temp)
    temp_minus = 1 - temp

    print(1-temp.mean())
    print(temp_minus.mean())
    # mask_path = '/home/bh/AAA/3d_unet_AAA/data/preprocess/mask_256/'
    # file_list = natsort.natsorted(os.listdir(mask_path))
    #
    # pos = 0
    # total = 0
    # fold = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
    # for idx in file_list:
    #     if (int(idx.split("_")[0]) not in fold):
    #         total = total + 1
    #         numpy_list = np.load(os.path.join(mask_path, idx))
    #         if(2==len(np.unique(numpy_list))):
    #             pos = pos + 1
    #
    # print(total)
    # print(pos)