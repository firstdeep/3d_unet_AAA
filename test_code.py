import os
import natsort
import time
import numpy as np

def DiceLoss(output, gt):

    epsilon = 1.
    print("==")
    print(output * gt)
    print(len(output.shape)-1)
    axes = tuple(range(1, len(output.shape)-1))
    numerator = 2 * np.sum(output * gt, (2,3))
    denominator = np.sum(np.square(output) + np.square(gt), (2,3))

    # return 1 - np.mean((numerator + epsilon) / (denominator + epsilon))


    # smooth = 1.
    # logits = output.flatten()
    # targets = gt.flatten()
    # numerator = np.sum(output * gt, (0,1,2,3))

    intersection = (logits * targets).sum()

    return 1 - (((2. * intersection) + smooth) /
                (logits.sum() + targets.sum() + smooth))

if __name__ =="__main__":


    gt = np.zeros((2,2,2,2))
    gt[0,0,0,0] = 1
    gt[0,0,0,1] = 1
    gt[0,1,0,1] = 1
    gt[0,1,1,1] = 1
    gt[1,0,0,0] = 1
    gt[1,0,0,1] = 1
    gt[1,1,0,0] = 1
    gt[1,1,1,0] = 1
    print(gt)
    print("====")
    output = np.random.randint(0,2, (2,2,2,2))
    print(output)
    print(gt)

    dice = DiceLoss(output, gt)

    dice1= DiceLoss(output[0], gt[0])
    dice2 = DiceLoss(output[1], gt[1])
    dice_mean = (dice1 + dice2) / 2.0
    print("==")