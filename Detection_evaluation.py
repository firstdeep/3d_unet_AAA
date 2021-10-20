import numpy as np
import os
import cv2
import natsort
import yaml


def load_config_yaml(config_file):
   return yaml.safe_load(open(config_file, 'r'))


def make_gt(path, subject_np):
    mask_path = path

    data_arr = np.zeros((len(subject_np), 3))  # 0: number of mask images about subject, 1: mask start point, 2. mask finish point
    ratio_pos = []
    subject_idx = subject_np

    count = 0  # mask count
    total_count = 0  # subject count

    file_pos_list = []
    total_pos_num_list = []

    for sub_idx in subject_idx:
        print(sub_idx)
        sub_idx = str(sub_idx)
        file_list = natsort.natsorted(os.listdir(os.path.join(mask_path, sub_idx)))
        for file_idx in file_list:
            num = int(file_idx.split('.')[0])

            total_count = total_count + 1
            mask = cv2.imread(os.path.join(mask_path, sub_idx, file_idx))
            # Check positive sample
            pos_check = int(len(np.unique(mask)))
            if pos_check == 2:
                count = count + 1
                file_pos_list.append(num)

        if (file_pos_list[-1] - file_pos_list[0] + 1) != len(file_pos_list):
            print("=== ERROR ===")
            print(sub_idx)
            print("=== ERROR ===")
            break
        sub_idx = int(sub_idx)
        data_arr[(sub_idx - 1), 0] = total_count
        data_arr[(sub_idx - 1), 1] = file_pos_list[0]
        data_arr[(sub_idx - 1), 2] = file_pos_list[-1]
        total_pos_num_list.append(count)
        ratio_pos.append(((file_pos_list[-1]-file_pos_list[0]+1)/total_count)*100)

        count = 0
        total_count = 0
        file_pos_list = []

    np.save("./GT_pos.npy", data_arr)
    print("*" * 50)
    print(data_arr)
    print(total_pos_num_list)
    print(sum(total_pos_num_list))
    print(ratio_pos)
    print(sum(ratio_pos)/len(ratio_pos))
    print("==== Done ====")

def make_pred(path, subject_np):
    ###############################################################
    # Prediction
    mask_path = path

    file_list = natsort.natsorted(os.listdir(mask_path))
    subject_idx = subject_np

    data_arr = np.zeros((len(subject_idx), 400))

    count = 0

    for sub_idx in subject_idx:
        ellipse = []
        print(sub_idx)
        for file_idx in file_list:

            idx_split = file_idx.split('_')
            num = int(idx_split[1].split('.')[0])

            if int(idx_split[0]) == sub_idx:
                mask = cv2.imread(os.path.join(mask_path, file_idx), cv2.IMREAD_GRAYSCALE)
                # Check positive sample
                pos_check = int(len(np.unique(mask)))

                if pos_check == 2:
                    # print(file_idx)
                    # if file_idx == "1_0070.png" or file_idx == "2_0067.png":
                    #     continue
                    # rect, ellip_diameter = fit_ellipse(mask)
                    # ### ellipse check
                    # ellipse.append(ellip_diameter)
                    #
                    # if ellip_diameter >= 18:
                    data_arr[(sub_idx - 1), count] = 1

                count = count + 1

        count = 0

    np.save("./pred_pos.npy", data_arr)
    print("*" * 50)
    print("==== Done ====")


if __name__ == "__main__":
    #####################
    #       MAIN        #
    #####################
    config_file_path = "./config/train_config.yaml"
    config = load_config_yaml(config_file_path)

    # Change subject Range
    # original 1~61 / [Now]10.13 1~15
    subject_idx = np.arange(1, 16)
    # subject_idx = np.arange(1, 61)

    mask_path =os.path.join( config['aaa']['file_path'], config['aaa']['mask_path'])
    make_gt(mask_path, subject_idx)

    mask_path = '/home/bh/PycharmProjects/3d_pytorch_bh/visualization/test_result_mask/'
    make_pred(mask_path, subject_idx)

    # real world We don't know GT value
    # So we make function and check
    gt_data = np.load("./GT_pos.npy")
    pred_data = np.load("./pred_pos.npy")

    total_rate = []
    total_fn = []
    total_fp = []

    mini_slice_num = 10

    for sub_idx in subject_idx:

        flag = 0

        sub_total_len = int(gt_data[(sub_idx - 1), 0])
        sub_start = int(gt_data[(sub_idx - 1), 1]) -1
        sub_finish = int(gt_data[(sub_idx - 1), 2]) -1
        num_gt = sub_finish - sub_start + 1

        gt_np = np.zeros(sub_total_len)
        gt_np[sub_start:sub_finish + 1] = 1

        pred_np = pred_data[(sub_idx - 1), :sub_total_len]
        num_pred = int(pred_np.sum())
        if num_pred == 0:
            print("=== Model Do not detected anything %d ===" % (sub_idx))
            total_rate.append(0)
            total_fn.append(0)
            total_fp.append(0)
            continue
        ###############################
        #   preprocessing using pred  #
        ###############################
        pred_true_idx = np.array(np.where(pred_np > 0)[0])
        pred_seg_idx = []
        # print(pred_true_idx)

        # start_point = pred_true_idx[0]
        # pred_seg_idx.append(start_point)
        # temp_list = []
        #
        # for idx in pred_true_idx[1:]:
        #     if idx - start_point == 1:
        #         start_point = idx
        #         pred_seg_idx.append(idx)
        #
        #     else:
        #         start_point = idx
        #         if len(pred_seg_idx) >= mini_slice_num:
        #             if len(temp_list) == 0:
        #                 temp_list = pred_seg_idx
        #                 flag = 1
        #             else:
        #                 # if flag == 1:
        #                 #     temp_list.extend(pred_seg_idx)
        #                 #     sorted(temp_list)
        #                 #     flag = 0
        #                 #
        #                 # elif len(temp_list) < len(pred_seg_idx):
        #                 #     temp_list = pred_seg_idx
        #                 if len(temp_list) < len(pred_seg_idx):
        #                     temp_list = pred_seg_idx
        #
        #             pred_seg_idx = []
        #         else:
        #             pred_seg_idx = []
        #
        #         pred_seg_idx.append(start_point)
        #
        # # No more than 1 difference between start and finish
        # if len(temp_list) < len(pred_seg_idx):
        #     temp_list = pred_seg_idx
        # # print(temp_list)
        #
        # # Fill in the empty value between start and finish value
        # np_predict_list = np.arange(temp_list[0], temp_list[-1] + 1)
        # seg_idx = list(np_predict_list)
        #
        # not_seg_idx = [x for x in range(len(pred_np)) if x not in seg_idx]
        # pred_np[not_seg_idx] = 0
        # pred_np[seg_idx] = 1

        # predict rate in GT
        pred_rate = pred_np[sub_start:sub_finish + 1].sum() / num_gt * 100

        fn = 100 - pred_rate

        pred_np[sub_start:sub_finish + 1] = 0
        fp = pred_np.sum() / num_pred * 100

        total_rate.append(pred_rate)
        total_fn.append(fn)
        total_fp.append(fp)

        print("\n=== Subject\"%d\" ===" % (sub_idx))
        print("GT_total_num = %d, GT_number of mask = %d, GT_start_point = %d, GT_finish_point = %d" % (
        sub_total_len, num_gt, sub_start, sub_finish))
        # print(
        #     "pre_total_num = %d, pre_number of mask = %d, pre_start_point = %d, pre_finish_point = %d, pre_num_of_mask_no_filtering = %d" % (
        #     sub_total_len, (temp_list[-1] - temp_list[0] + 1), temp_list[0], temp_list[-1], len(pred_true_idx)))
        print("False negative rate = %.2f%% , False positive rate = %.2f%%" % (fn, fp))
        print("[Predict(in GT) / GT range] rate = %.2f%%\n\n" % (pred_rate))
        # print("%.2f"%fn)

    print("total_rate = %.2f%%, total_fn = %.2f%%, total_fp = %.2f%%" % (
    (sum(total_rate) / len(total_rate)), (sum(total_fn) / len(total_fn)), (sum(total_fp) / len(total_fp))))
