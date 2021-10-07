from unet_3d_model import UNet3D
import torch
import torch.nn as nn
import yaml
import cv2

from utils import *
from loss_func import *
from preprocessing_input_data import pre_data_saving


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
        print("Model parameter num: %d"%(count_parameter(model)))
        print(model)

        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.RMSprop(params, lr=config['optimizer']['lr'])

        loss_criterion = nn.BCEWithLogitsLoss()
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.99)

        print("=============================")
        print("========== FOLD %d ==========" % fold)
        print("=============================")

        if config['trainer']['mode'] == 'train':

            model.to(device)

            # 10.07: Change to use less memory
            loaders = get_aaa_train_loader(config, train_ids)

            for epoch in range(0,config['trainer']['max_epochs']):
                print("** Epoch %d"%(epoch+1))
                model.train()
                loss_sum = []
                for t in loaders['train']:
                    input, target, weight = split_training_batch(t, device)
                    output = model(input)

                    loss = loss_criterion(output, target)
                    loss_sum.append(loss.item())

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                lr_scheduler.step()

                print("** [INFO] Loss = %f & LR = %f"%(sum(loss_sum)/len(loss_sum), lr_scheduler.get_last_lr()[0]))

                torch.save(model.state_dict(), './pretrained/3d_deepAAA_%d.pth'%fold)
            ########################################################################################
            # validation
            if epoch % 10 == 0:
                print(epoch)
                # Evaluation

            print("=== Training Done")

        ########################################################################################
        # EVALUATION
        if config['trainer']['mode'] == 'train' =='test':

            loaders = get_aaa_test_loader(config, train_ids)

            count = 1
            model.load_state_dict(torch.load('./pretrained/3d_unet_256_36_bce_0.0001.pth'))
            model.to(device)
            model.eval()

            for t in loaders['train']:
                for idx in range(len(t[0])):

                    input = t[0][idx]
                    label = t[1][idx]
                    input = input.to(device)

                    output = model(input)
                    sig = nn.Sigmoid()
                    output = sig(output)

                    output = output[0,0].detach().cpu().numpy() * 255.
                    input = input[0,0].detach().cpu().numpy() * 255.

                    output[output > 127.5] = 255
                    output[output < 127.5] = 0

                    output = output.astype(np.uint8)
                    input = input.astype(np.uint8)

                    for i in range(0,24):
                        cv2.imwrite("./result/raw/%d_%d.png"%(count, 24*idx+i), input[0])
                        cv2.imwrite("./result/mask/%d_%d.png"%(count, 24*idx+i), output[0])


                count = count + 1





if __name__ =="__main__":

    config_file_path = "./config/train_config.yaml"
    config = load_config_yaml(config_file_path)

    # pre_data_saving(config=config)

    main(config)
