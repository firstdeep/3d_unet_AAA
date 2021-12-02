import os.path

from unet_3d_model import UNet3D
import torch
import torch.nn as nn
import yaml
import cv2
import time
import shutil

from tqdm import tqdm
from utils import *
from loss_func import *
from evaluation import *
from preprocessing_input_data import pre_train_data_saving, pre_test_data_saving

import volumentations as vol

def load_config_yaml(config_file):
   return yaml.safe_load(open(config_file, 'r'))


def initialize_weights(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv3d') != -1:
        nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)

    elif classname.find('BatchNorm') != -1:
        nn.init.constant_(m.weight.data, 1)
        nn.init.constant_(m.bias.data, 0)



def main(config):

    total_subject = list(range(1,54))
    kfold = KFold(n_splits=4, shuffle=False)

    for fold, (train_ids, test_ids) in enumerate(kfold.split(total_subject)):
        if fold!=0:
            continue
        torch.cuda.empty_cache()

        for index, value in enumerate(test_ids):
           test_ids[index] = value + 1
        for index, value in enumerate(train_ids):
           train_ids[index] = value + 1


        GPU_NUM = config['trainer']['gpu_idx']
        device = torch.device(f'cuda:{GPU_NUM}') if torch.cuda.is_available() else torch.device('cpu')

        model = UNet3D(n_channels=1, n_classes=1)
        model.apply(initialize_weights)

        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.RMSprop(params, lr=config['optimizer']['lr'])
        print("**Learning Rate: %f"%config['optimizer']['lr'])
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.99)

        if config['loss']['name'] =="BCEWithLogitsLoss":
            print("BCEWithLogitsLoss")
            loss_criterion = nn.BCEWithLogitsLoss()
        elif config['loss']['name'] == "dice":
            print("** Dice loss **")
            # loss_criterion = DiceLoss_bh()
            loss_criterion = DiceLoss()
            print(loss_criterion)

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

            if config['aaa']['transform']:
                train_transform = vol.Compose([
                    vol.RotatePseudo2D(axes=(0,1), limit=(-10,10), interpolation=1)
                    # ,vol.ElasticTransformPseudo2D(alpha=1, sigma=10, alpha_affine=10)
                ])
            else: train_transform = None

            loaders = get_aaa_train_loader(config, train_ids, transform=train_transform)

            if not config['trainer']['resume']:
                resume_epoch = 0

            for epoch in range(resume_epoch,config['trainer']['max_epochs']+1):

                start_time = time.time()
                model.train()
                loss_sum = []

                for i, t in tqdm(enumerate(loaders['train']), desc="[Epoch %d]Training..."%(epoch)):
                    print(" ", end='\r')
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
                # print(loss_sum)
                print("** [INFO] Epoch \"%d\", Loss = %f & LR = %f & Time = %.2f min" % (
                epoch, sum(loss_sum)/len(loss_sum), lr_scheduler.get_last_lr()[0], ((time.time() - start_time) / 60.)))

                if epoch % 10 == 0:
                    file_name = config['trainer']['save_valid_name']
                    torch.save({'epoch': epoch,
                               'model_state_dict': model.state_dict(),
                               'optimizer_state_dict': optimizer.state_dict()
                               }, './pretrained/%s_epoch%d_%d.pth'%(file_name,epoch,fold))

                ########################################################################################

                # validation
                if config['trainer']['validation']:

                    if epoch % config['trainer']['validate_after_iters'] == 0:
                        valid_idx = int(epoch // config['trainer']['validate_after_iters'])

                        model.eval()

                        s_sum, t_sum = 0, 0
                        intersection, union = 0, 0
                        s_diff_t, t_diff_s = 0, 0

                        for i, t in enumerate(loaders['val']):
                            input, target = split_training_batch_validation(t, device)
                            # input & output shape: batch*1*8(slice_num)*256*256
                            output = model(input)

                            sig = nn.Sigmoid()
                            output_sig = sig(output)

                            pred = np.array(output_sig.data.cpu())
                            target = np.array(target.data.cpu())

                            s_sum, t_sum, s_diff_t, t_diff_s, intersection, union = eval_segmentation_volume(config, pred, target, input, idx=i, validation_idx=valid_idx)
                            s_sum += s_sum
                            t_sum += t_sum
                            s_diff_t += s_diff_t
                            t_diff_s += t_diff_s
                            intersection += intersection
                            union += union

                        overlap = intersection / t_sum
                        jaccard = intersection / union
                        dice = 2.0 * intersection / (s_sum + t_sum)
                        fn = t_diff_s / t_sum
                        fp = s_diff_t / s_sum

                        print('** [INFO] Validation_%d Evaluation: Overlap: %.4f, Jaccard: %.4f, Dice: %.4f, FN: %.4f, FP: %.4f\n'
                          % (valid_idx, overlap, jaccard, dice, fn, fp))

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

            path = os.path.join(config['aaa']['file_path'], config['aaa']['mask_path'])

            folder_idx = natsort.natsorted(os.listdir(path))
            for fidx in folder_idx:
                file_idx = natsort.natsorted(os.listdir(os.path.join(path, fidx)))
                subject_depth.append(len(file_idx))

            # Pretrained model Loading...
            path = config['trainer']['save_model_path']
            file_name = config['trainer']['save_model_name'] + "_%d.pth" % fold
            print("Loading pretrained model: %s" % (os.path.join(path, file_name)))
            state = torch.load(os.path.join(path, file_name), map_location=device)
            model.load_state_dict(state['model_state_dict'])
            model.eval()
            model.to(device)

            loaders = get_aaa_test_loader(config, test_ids)

            valid_path = config['aaa']['validation_path']
            valid_folder = "test_result"
            valid_mask_folder = "test_result_mask"
            if os.path.exists(os.path.join(valid_path, valid_folder)):
                shutil.rmtree(os.path.join(valid_path, valid_folder))
                shutil.rmtree(os.path.join(valid_path, valid_mask_folder))


            for i, t in enumerate(loaders['test']):
                input, target, idx = split_training_batch_validation(t, device)
                # input & output shape: batch*1*8(slice_num)*256*256
                output = model(input)

                sig = nn.Sigmoid()
                output_sig = sig(output)

                pred = np.array(output_sig.data.cpu())
                target = np.array(target.data.cpu())

                eval_segmentation_visualization(config, pred, target, input, file_name=idx[0], sub_depth=subject_depth)

            subj_ol = []
            subj_ja = []
            subj_di = []
            subj_fn = []
            subj_fp = []

            for subject in test_ids:
                overlap, jaccard, dice, fn, fp = eval_segmentation_volume_test(config, "./visualization/test_result_mask", str(subject))
                print(str(subject) + ' %.4f %.4f %.4f %.4f %.4f' % (overlap, jaccard, dice, fn, fp))
                subj_ol.append(overlap)
                subj_ja.append(jaccard)
                subj_di.append(dice)
                subj_fn.append(fn)
                subj_fp.append(fp)
            # del subj_ol[5:6]
            # del subj_ja[5:6]
            # del subj_di[5:6]
            # del subj_fn[5:6]
            # del subj_fp[5:6]
            print('** [INFO] Overlap: %.4f, Jaccard: %.4f, Dice: %.4f, FN: %.4f, FP: %.4f\n'
                % (np.mean(subj_ol), np.mean(subj_ja), np.mean(subj_di), np.mean(subj_fn), np.mean(subj_fp)))


if __name__ =="__main__":

   config_file_path = "./config/train_config.yaml"
   config = load_config_yaml(config_file_path)

   # pre_train_data_saving(config=config)
   # pre_test_data_saving(config=config)

   main(config)