import os.path

from unet_3d_model import UNet3D
import torch
import torch.nn as nn
import yaml
import cv2
import time

from utils import *
from loss_func import *
from evaluation import *
from preprocessing_input_data import pre_train_data_saving, pre_test_data_saving


def load_config_yaml(config_file):
   return yaml.safe_load(open(config_file, 'r'))


def main(config):

    total_subject = list(range(1,61))
    kfold = KFold(n_splits=4, shuffle=False)

    for fold, (train_ids, test_ids) in enumerate(kfold.split(total_subject)):

        for index, value in enumerate(test_ids):
           test_ids[index] = value + 1
        for index, value in enumerate(train_ids):
           train_ids[index] = value + 1


        GPU_NUM = config['trainer']['gpu_idx']
        device = torch.device(f'cuda:{GPU_NUM}') if torch.cuda.is_available() else torch.device('cpu')

        model = UNet3D(n_channels=1, n_classes=1)
        # print("Model parameter num: %d"%(count_parameter(model)))
        # print(model)

        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.RMSprop(params, lr=config['optimizer']['lr'])

        if config['loss']['name'] =="BCEWithLogitsLoss":
            loss_criterion = nn.BCEWithLogitsLoss()
        elif config['loss']['name'] == "dice":
            loss_criterion = DiceLoss()

        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.99)

        print("=============================")
        print("========== FOLD %d ==========" % fold)
        print("=============================")

        if config['trainer']['mode'] == 'train':
            # If pretrained exist, load model & optimizer
            resume_epoch = 0
            if config['trainer']['resume']:
                path = config['trainer']['save_model_path']
                file_name = config['trainer']['save_model_name']+"_%d.pth"%fold
                print("Loading pretrained model: %s" % (os.path.join(path, file_name)))
                state = torch.load(os.path.join(path, file_name), map_location=device)
                model.load_state_dict(state['model_state_dict'])
                optimizer.load_state_dict(state['optimizer_state_dict'])
                resume_epoch = state['epoch']

                # Optimizer device setting
                for state in optimizer.state.values():
                   for k, v in state.items():
                       if isinstance(v, torch.Tensor):
                           state[k] = v.to(device)

            model.to(device)
            # 21.10.07: Change to use less memory
            loaders = get_aaa_train_loader(config, train_ids)

            for epoch in range(0,config['trainer']['max_epochs']):

                if config['trainer']['resume']:
                    epoch = resume_epoch

                start_time = time.time()
                model.train()
                loss_sum = []

                for t in loaders['train']:
                    input, target, weight = split_training_batch(t, device)
                    output = model(input)

                    if config['loss']['name'] == "BCEWithLogitsLoss":
                        loss = loss_criterion(output, target)
                    elif config['loss']['name'] == "dice":
                        sig = nn.Sigmoid()
                        loss = loss_criterion(sig(output), target)

                    loss_sum.append(loss.item())

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                lr_scheduler.step()

                print("** [INFO] Epoch \"%d\", Loss = %f & LR = %f & Time = %.2f min" % (
                epoch, sum(loss_sum)/len(loss_sum), lr_scheduler.get_last_lr()[0], ((time.time() - start_time) / 60.)))

                torch.save({'epoch': epoch,
                           'model_state_dict': model.state_dict(),
                           'optimizer_state_dict': optimizer.state_dict()
                           }, './pretrained/3d_deepAAA_70_%d.pth'%fold)

                ########################################################################################
                # validation
                if epoch % config['trainer']['validate_after_iters'] == 0:
                    valid_idx = int(epoch // config['trainer']['validate_after_iters'])

                    model.eval()

                    valid_over = []
                    valid_jaccard = []
                    valid_dice = []
                    valid_fn = []
                    valid_fp = []

                    for i, t in enumerate(loaders['val']):
                        input, target = split_training_batch_validation(t, device)
                        # input & output shape: batch*1*8(slice_num)*256*256
                        output = model(input)

                        sig = nn.Sigmoid()
                        output_sig = sig(output)

                        pred = np.array(output_sig.data.cpu())
                        target = np.array(target.data.cpu())

                        overlap, jaccard, dice, fn, fp = eval_segmentation_volume(config, pred, target, input, idx=i, validation_idx=valid_idx)
                        valid_over.append(overlap)
                        valid_jaccard.append(jaccard)
                        valid_dice.append(dice)
                        valid_fn.append(fn)
                        valid_fp.append(fp)

                    print('** [INFO] Validation_%d Evaluation: Overlap: %.4f, Jaccard: %.4f, Dice: %.4f, FN: %.4f, FP: %.4f\n'
                          % (valid_idx, np.mean(valid_over), np.mean(valid_jaccard), np.mean(valid_dice), np.mean(valid_fn), np.mean(valid_fp)))


            print("=== Epoch iteration done ===")

        ########################################################################################
        # Test datasets evaluation
        if config['trainer']['mode'] =='test':

            print("*******************")
            print("     Testing...    ")
            print("     FOLD \"%d\"   "%(fold))
            print("*******************")

            # subject_depth
            subject_depth = []
            path = './data/mask_256_pos'
            folder_idx = natsort.natsorted(os.listdir(path))
            for fidx in folder_idx:
                file_idx = natsort.natsorted(os.listdir(os.path.join(path, fidx)))
                subject_depth.append(len(file_idx))

            # Pretrained model Loading...
            path = config['trainer']['save_model_path']
            file_name = config['trainer']['save_model_name'] + "_%d.pth" % fold
            state = torch.load(os.path.join(path, file_name), map_location=device)
            model.load_state_dict(state['model_state_dict'])
            model.eval()
            model.to(device)

            loaders = get_aaa_test_loader(config, test_ids)

            test_over = []
            test_jaccard = []
            test_dice = []
            test_fn = []
            test_fp = []

            for i, t in enumerate(loaders['test']):
                input, target, idx = split_training_batch_validation(t, device)
                # input & output shape: batch*1*8(slice_num)*256*256
                output = model(input)

                sig = nn.Sigmoid()
                output_sig = sig(output)

                pred = np.array(output_sig.data.cpu())
                target = np.array(target.data.cpu())

                overlap, jaccard, dice, fn, fp = eval_segmentation_volume_test(config, pred, target, input,
                                                                          file_name=idx[0], sub_depth=subject_depth)

                test_over.append(overlap)
                test_jaccard.append(jaccard)
                test_dice.append(dice)
                test_fn.append(fn)
                test_fp.append(fp)

            print('** [INFO] Overlap: %.4f, Jaccard: %.4f, Dice: %.4f, FN: %.4f, FP: %.4f\n'
                % (np.mean(test_over), np.mean(test_jaccard), np.mean(test_dice), np.mean(test_fn), np.mean(test_fp)))




if __name__ =="__main__":

   config_file_path = "./config/train_config.yaml"
   config = load_config_yaml(config_file_path)

   # pre_train_data_saving(config=config)
   # pre_test_data_saving(config=config)

   main(config)
