import os
import natsort

if __name__ =="__main__":
    path = './data/mask_256_pos'

    folder_idx = natsort.natsorted(os.listdir(path))
    count = 1

    for fidx in folder_idx:
        file_idx = natsort.natsorted(os.listdir(os.path.join(path, fidx)))
        print("%d. %d"%(count, len(file_idx)))
        final_png = file_idx[-1]
        final_num = int(final_png.split('.')[0])
        if final_num!=len(file_idx):
            print("*** Not Matching ***")
            print("== %s =="%fidx)

        count = count + 1