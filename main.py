from unet_3d_model import UNet3D
import torch
import torch.nn as nn
import yaml
from utils import *
import cv2

def count_parameter(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class NegDiceLoss(nn.Module):
    def __init__(self):
        super(NegDiceLoss, self).__init__()
    def forward(self, logits, targets):
        smooth = 1.
        logits = torch.flatten(logits)
        targets = torch.flatten(targets)
        intersection = (logits * targets).sum()

        negative_dice = 1 -(((2. * intersection) + smooth) / (logits.sum() + targets.sum() + smooth))
        return negative_dice


def _split_training_batch(t, device):
    def _move_to_device(input):
        if isinstance(input, tuple) or isinstance(input, list):
            return tuple([_move_to_device(x) for x in input])
        else:
            return input.to(device)

    t = _move_to_device(t)
    weight = None
    if len(t) == 2:
        input, target = t
    else:
        input, target, weight = t
    return input, target, weight



def _load_config_yaml(config_file):
    return yaml.safe_load(open(config_file, 'r'))



def main(gpu_idx, mode, lr = 0.001, epoch_max=50):

    config_file_path = "./config/train_config.yaml"

    config = _load_config_yaml(config_file_path)

    GPU_NUM = gpu_idx
    device = torch.device(f'cuda:{GPU_NUM}') if torch.cuda.is_available() else torch.device('cpu')

    model = UNet3D(n_channels=1, n_classes=1)

    print(count_parameter(model))

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.RMSprop(params, lr=lr)
    # loss_criterion = NegDiceLoss()
    loss_criterion = nn.BCEWithLogitsLoss()
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.99)

    loaders = get_aaa_train_loader(config)

    if mode == 'train':

        model.to(device)
        for epoch in range(0,epoch_max):
            print("=== Epoch %d ==="%(epoch+1))
            model.train()
            loss_list = []
            for t in loaders['train']:
                input, target, weight = _split_training_batch(t, device)
                output = model(input)
                # output = output[:,1:]

                loss = loss_criterion(output, target)
                print(loss)
                # loss_list.append(loss)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            lr_scheduler.step()
            print(lr_scheduler.get_last_lr())
            torch.save(model.state_dict(), './pretrained/3d_unet_256_36_bce_0.0001.pth')
            # print(sum(loss)/len(loss))
        print("=== Training Done")


    if mode =='test':
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
    print("=== Training Start")
    lr = 0.0001
    epoch_max = 1000
    gpu_idx = 2
    mode = 'test'
    main(gpu_idx, mode, lr=lr, epoch_max=epoch_max)
