import numpy as np
import cv2
import os
from PIL import Image
import natsort

def eval_segmentation_volume(config, pred, target, input_img, idx=0, validation_idx=0):

    """
    Calculate only one person 3D volume evaluation function
    pred & target shape: Batch_size * 1 * Slice * H * W
    pred: 0 or 1 (np.uint32) after sigmoid function
    target: 0 or 1 (np.uint32)
    """

    input_img = np.array(input_img.data.cpu()) * 255
    pred[pred>0.5] = 1.
    pred[pred<=0.5] = 0.

    # calculation
    s_sum, t_sum = 0, 0
    intersection, union = 0, 0
    s_diff_t, t_diff_s = 0, 0

    batch_over = []
    batch_jaccard = []
    batch_dice = []
    batch_fn = []
    batch_fp = []

    for batch in range (0, pred.shape[0]):
        batch_pred = pred[batch][0]
        batch_target = target[batch][0]
        batch_input = input_img[batch][0]

        for slice in range(0,batch_pred.shape[0]):
            pred_slice = batch_pred[slice].astype(np.uint32)
            gt_slice = batch_target[slice].astype(np.uint32)
            input_slice = batch_input[slice].astype(np.uint8)

            s_sum += pred_slice.sum()
            t_sum += gt_slice.sum()

            intersection += np.bitwise_and(pred_slice, gt_slice).sum()
            union += np.bitwise_or(pred_slice, gt_slice).sum()

            # prediction (green)
            s_diff_t += (pred_slice - np.bitwise_and(pred_slice, gt_slice)).sum()
            # target (red)
            t_diff_s += (gt_slice - np.bitwise_and(pred_slice, gt_slice)).sum()

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
            valid_folder = config['trainer']['save_valid_name']+"_validation_%d"%validation_idx
            if not os.path.exists(os.path.join(valid_path, valid_folder)):
                os.mkdir(os.path.join(valid_path, valid_folder))
            valid_name = "val_%d_batch_%d_slice_%d.png"%(idx, batch, slice)
            cv2.imwrite(os.path.join(valid_path,valid_folder,valid_name), img_all)
            ############################################################################################################

        # overlap = intersection / t_sum
        # jaccard = intersection / union
        # dice = 2.0*intersection / (s_sum + t_sum)
        # fn = t_diff_s / t_sum
        # fp = s_diff_t / s_sum

    return s_sum, t_sum, s_diff_t, t_diff_s, intersection, union


def eval_segmentation_volume_test_all(config, save_path="", subj_id=0):
    # mask file load
    pred_path = save_path
    gt = np.load(os.path.join(config['aaa']['prepro_path'], config['aaa']['mask_path'], "%s.npy"%subj_id))

    pred_mask_list = natsort.natsorted([name for name in os.listdir(pred_path) if subj_id == name.split("_")[0]])

    # print("[Subject = \"%d\"] & number of pred image \"%d\" & num of GT \"%d\" "%(int(subject), len(pred_mask_list), len(gt_mask_list)))

    # calculation
    s_sum, t_sum = 0, 0
    intersection, union = 0, 0
    s_diff_t, t_diff_s = 0, 0

    if(gt.shape[0] == len(pred_mask_list)):
        for i in range(len(pred_mask_list)):

            gt_slice = (gt[i]/255.0).astype(np.uint32)

            pred_slice = Image.open(os.path.join(pred_path, pred_mask_list[i]))
            pred_slice = (np.array(pred_slice) / 255.0).astype(np.uint32)

            s_sum += pred_slice.sum()
            t_sum += gt_slice.sum()

            intersection += np.bitwise_and(pred_slice, gt_slice).sum()
            union += np.bitwise_or(pred_slice, gt_slice).sum()


            s_diff_t += (pred_slice - np.bitwise_and(pred_slice, gt_slice)).sum()
            t_diff_s += (gt_slice - np.bitwise_and(pred_slice, gt_slice)).sum()


        overlab = intersection / t_sum
        jaccard = intersection / union
        dice = 2.0*intersection / (s_sum + t_sum)
        fn = t_diff_s / t_sum
        fp = s_diff_t / s_sum

        return overlab, jaccard, dice, fn, fp

def eval_segmentation_volume_test(config, save_path="", subj_id=0):
    # mask file load
    pred_path = save_path
    gt = np.load(os.path.join(config['aaa']['prepro_path'], "mask_128", "%s.npy"%subj_id))
    pred_mask_list = natsort.natsorted([name for name in os.listdir(pred_path) if subj_id == name.split("_")[0]])
    gt_list = natsort.natsorted(os.listdir("/home/bh/AAA/3d_unet_AAA/data_1227/mask_256/"))

    # calculation
    s_sum, t_sum = 0, 0
    intersection, union = 0, 0
    s_diff_t, t_diff_s = 0, 0

    if(gt.shape[0] == len(pred_mask_list)):
        for i in range(len(pred_mask_list)):

            gt_slice = cv2.imread(os.path.join("/home/bh/AAA/3d_unet_AAA/data_1227/mask_256/",pred_mask_list[i]), cv2.IMREAD_GRAYSCALE)
            gt_slice = (gt_slice/255.0).astype(np.uint32)

            pred_slice = Image.open(os.path.join(pred_path, pred_mask_list[i]))
            pred_slice = (np.array(pred_slice) / 255.0).astype(np.uint32)

            s_sum += pred_slice.sum()
            t_sum += gt_slice.sum()

            intersection += np.bitwise_and(pred_slice, gt_slice).sum()
            union += np.bitwise_or(pred_slice, gt_slice).sum()


            s_diff_t += (pred_slice - np.bitwise_and(pred_slice, gt_slice)).sum()
            t_diff_s += (gt_slice - np.bitwise_and(pred_slice, gt_slice)).sum()


        overlab = intersection / t_sum
        jaccard = intersection / union
        dice = 2.0*intersection / (s_sum + t_sum)
        fn = t_diff_s / t_sum
        fp = s_diff_t / s_sum

        return overlab, jaccard, dice, fn, fp

