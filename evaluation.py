import numpy as np


def eval_segmentation_volume(pred, target):
    """
    Calculate only one person 3D volume evaluation function
    pred & target shape: Batch_size * 1 * Slice * H * W
    pred: 0 or 1 (np.uint32) after sigmoid function
    target: 0 or 1 (np.uint32)
    """

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
        for slice in range(0,batch_pred.shape[0]):
            pred_slice = batch_pred[slice].astype(np.uint32)
            gt_slice = batch_target[slice].astype(np.uint32)

            s_sum += pred_slice.sum()
            t_sum += gt_slice.sum()

            intersection += np.bitwise_and(pred_slice, gt_slice).sum()
            union += np.bitwise_or(pred_slice, gt_slice).sum()

            # prediction (green)
            s_diff_t += (pred_slice - np.bitwise_and(pred_slice, gt_slice)).sum()
            # target (red)
            t_diff_s += (gt_slice - np.bitwise_and(pred_slice, gt_slice)).sum()

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
        # print('Subject:%d / %.4f %.4f %.4f %.4f %.4f' % (batch, overlap, jaccard, dice, fn, fp))

    return np.mean(batch_over), np.mean(batch_jaccard), np.mean(batch_dice), np.mean(batch_fn), np.mean(batch_fp)
