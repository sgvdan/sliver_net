import torch
from torch.nn import Conv2d
from torchvision.transforms import InterpolationMode

from kermany_data import KermanyDataset
import sys
sys.path.append('/home/projects/ronen/sgvdan/workspace/projects/sliver_net')
import torchvision
from train import train, train_loop, evaluate
import wandb
from kermany_data import KERMANY_LABELS
from sliver_net import data
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--mode', type=int)
args = parser.parse_args()
mode = args.mode

wandb.login()
wandb.init()


resize_transform = torchvision.transforms.Resize((256, 256))
mask_resize_transform = torchvision.transforms.Resize((256, 256), InterpolationMode.NEAREST)


resnet18 = torchvision.models.resnet18(num_classes=4, pretrained=False)
if mode == 0:
    print('image', flush=True)
    resnet18.conv1 = Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
elif mode == 1:
    print('mask, image', flush=True)
    resnet18.conv1 = Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
elif mode == 2:
    print('channel, channel, ..., channel, image', flush=True)
    resnet18.conv1 = Conv2d(11, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
elif mode == 3:
    print('zeros, image', flush=True)
    resnet18.conv1 = Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    # resnet18.conv1.weight.data.fill_(0)
    # print('ALL WEIGHTS OF CONV1 ARE SET TO ZERO.')
elif mode == 4:
    print('mask(foreground), image', flush=True)
    resnet18.conv1 = Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
elif mode == 5:
    print('channel(foreground), channel(foreground), ..., channel(foreground), image', flush=True)
    resnet18.conv1 = Conv2d(9, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
elif 6 <= mode <= 15:
    print('single channel: {}'.format(mode-6))
    resnet18.conv1 = Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
else:
    print('Unsupported', flush=True)
    exit(0)

resnet18 = resnet18.cuda()

criterion = torch.nn.functional.cross_entropy
optimizer = torch.optim.Adam(resnet18.parameters(), lr=1e-5)

data.LABELS = KERMANY_LABELS

# (https://www.kaggle.com/paultimothymooney/kermany2018)
val_dataset = KermanyDataset('/home/projects/ronen/sgvdan/workspace/datasets/kermany/original/val',
                             '/home/projects/ronen/sgvdan/workspace/datasets/kermany/layer-segmentation/val',
                             mode=mode, image_transform=resize_transform, mask_transform=mask_resize_transform)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=64, shuffle=True)

train_dataset = KermanyDataset('/home/projects/ronen/sgvdan/workspace/datasets/kermany/original/train',
                               '/home/projects/ronen/sgvdan/workspace/datasets/kermany/layer-segmentation/train',
                               mode=mode, image_transform=resize_transform, mask_transform=mask_resize_transform)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

# print('CHANGED IMAGE READ METHOD TO ToTensor()(Image.open(...))', flush=True)

for i in range(0, 10):
    train(resnet18, criterion, optimizer, train_loader, val_loader, 1, 'cuda', 'ResNet18-Kermany')

    # Save temporarily, so that the model won't disappear
    tmp_model_name = 'single-channel-backbones/mode-{}-epoch-{}.pth'.format(mode, i+1)
    torch.save({"model_state_dict": resnet18.state_dict(),
                "optimizer_state_dict": optimizer.state_dict()
                }, tmp_model_name)
    print('Saved {}'.format(tmp_model_name))
    test_dataset = KermanyDataset('/home/projects/ronen/sgvdan/workspace/datasets/kermany/original/test',
                                  '/home/projects/ronen/sgvdan/workspace/datasets/kermany/layer-segmentation/test',
                                  mode=mode, image_transform=resize_transform, mask_transform=mask_resize_transform)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=True)

    avg_iou, avg_accuracy = evaluate(resnet18, test_loader, 'Kermany2D/Test', device='cuda')
    print('TEST: Average IOU: {}, Accuracy:{}'.format(avg_iou, avg_accuracy))

print('Done Training.')
