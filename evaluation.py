import numpy as np
import cv2
import os

def eval_segmentation_volume(pred, target, input_img, idx=0, validation_idx=0):

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
            if not os.path.exists(
                    "/home/bh/PycharmProjects/3d_pytorch_bh/visualization/validation_%d/" % (validation_idx)):
                os.mkdir("/home/bh/PycharmProjects/3d_pytorch_bh/visualization/validation_%d/" % (validation_idx))

            cv2.imwrite("/home/bh/PycharmProjects/3d_pytorch_bh/visualization/validation_%d/val_%d_batch_%d_slice_%d.png"%(validation_idx, idx, batch, slice), img_all)
            ############################################################################################################

        overlap = intersection / t_sum
        jaccard = intersection / union
        dice = 2.0*intersection / (s_sum + t_sum)
        fn = t_diff_s / t_sum
        fp = s_diff_t / s_sum

        batch_over.append(overlap)
        batch_jaccard.append(jaccard)
        batch_dice.append(dice)
        batch_fn.append(fn)
        batch_fp.append(fp)

    return np.mean(batch_over), np.mean(batch_jaccard), np.mean(batch_dice), np.mean(batch_fn), np.mean(batch_fp)

