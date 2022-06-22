from tqdm.asyncio import tqdm
from playground.cache import Cache
from sliver_net.model import load_backbone
from sliver_net.model import SliverNet2
from sliver_net.data import build_volume_cache, variable_size_collate
from playground.cache import Cache
from tqdm.asyncio import tqdm
from playground.train import evaluate
from sliver_net import data
from playground.cache import Cache
from sliver_net.model import load_backbone
from sliver_net.model import SliverNet2
import torch
from sliver_net import data
from kermany_data import KermanyDataset
import sys
import torchvision
from train import train, train_loop
import wandb
from sliver_net.data import build_volume_cache, E2ETileDataset


def build_tile_dataset(build_from_scratch=False):
    volume_test_cache = Cache('volume_test_set')
    if build_from_scratch:
        build_volume_cache(volume_test_cache, '../../OCT-DL/Data/test/control', data.LABELS['HEALTHY'])
        build_volume_cache(volume_test_cache, '../../OCT-DL/Data/test/study', data.LABELS['SICK'])

    volume_train_cache = Cache('volume_train_set')
    if build_from_scratch:
        build_volume_cache(volume_train_cache, '../../OCT-DL/Data/train/control', data.LABELS['HEALTHY'])
        build_volume_cache(volume_train_cache, '../../OCT-DL/Data/train/study', data.LABELS['SICK'])

    volume_validation_cache = Cache('volume_validation_set')
    if build_from_scratch:
        build_volume_cache(volume_validation_cache, '../../OCT-DL/Data/validation/control', data.LABELS['HEALTHY'])
        build_volume_cache(volume_validation_cache, '../../OCT-DL/Data/validation/study', data.LABELS['SICK'])

    if build_from_scratch:
        print('Built tile dataset.')
    else:
        print('Loaded tile dataset.')

    return volume_train_cache, volume_validation_cache, volume_test_cache


def run():
    wandb.login()
    wandb.init()

    resize_transform = torchvision.transforms.Resize((256, 256))
    resnet18 = torchvision.models.resnet18(num_classes=4, pretrained=False).cuda()
    criterion = torch.nn.functional.cross_entropy
    optimizer = torch.optim.Adam(resnet18.parameters(), lr=1e-4)

    # Build loaders
    volume_train_cache, volume_validation_cache, volume_test_cache = build_tile_dataset()

    tile_train_dataset = E2ETileDataset(volume_train_cache, transform=resize_transform)
    tile_train_loader = torch.utils.data.DataLoader(dataset=tile_train_dataset, batch_size=1, shuffle=True,
                                                    collate_fn=variable_size_collate)

    tile_validation_dataset = E2ETileDataset(volume_validation_cache, transform=resize_transform)
    tile_validation_loader = torch.utils.data.DataLoader(dataset=tile_validation_dataset, batch_size=1, shuffle=True,
                                                         collate_fn=variable_size_collate)

    # Load Model
    backbone = load_backbone("sgvdan-kermany").cuda() # randomly initialize a resnet18 backbone
    sliver_model = SliverNet2(backbone, n_out=2).cuda() # create SLIVER-Net with n_out outputs
    print('SliverNet model Loaded!')

    # First: fine tune on everything but the backbone
    data.LABELS = {'HEALTHY': torch.nn.functional.one_hot(torch.tensor(0), 2), 'SICK': torch.nn.functional.one_hot(torch.tensor(1), 2)}

    # Fine Tune Last bit of SliverNet
    epochs = 3
    print('Fine tune on last bit of SliverNet for epochs={epochs}'.format(epochs=epochs))
    sliver_model.backbone.eval()
    for param in sliver_model.backbone.parameters():
        param.requires_grad = False
    optimizer = torch.optim.Adam(sliver_model.parameters(), lr=1e-4)
    train(sliver_model, sliver_model.loss_func, optimizer, tile_train_loader, tile_validation_loader, epochs, 'cuda', 'SliverNet/partial')

    # Fine Tune Whole SilverNet
    epochs = 7
    print('Fine tune on whole of SliverNet for epochs={epochs}'.format(epochs=epochs))
    sliver_model.backbone.train()
    for param in sliver_model.backbone.parameters():
        param.requires_grad = True
    optimizer = torch.optim.Adam(sliver_model.parameters(), lr=1e-5)
    train(sliver_model, sliver_model.loss_func, optimizer, tile_train_loader, tile_validation_loader, epochs, 'cuda', 'SliverNet/full')

    # Save model just in case
    tmp_model_name = './tmp_model.pth'
    torch.save({"model_state_dict": sliver_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict()
                }, tmp_model_name)
    print('Done. Saved model: {}'.format(tmp_model_name))

    # Evaluate on Hadassah Tile Set
    tile_test_dataset = E2ETileDataset(volume_test_cache, transform=resize_transform)
    tile_test_loader = torch.utils.data.DataLoader(dataset=tile_test_dataset, batch_size=30, shuffle=True,
                                                   collate_fn=variable_size_collate)

    avg_iou, avg_accuracy = evaluate(sliver_model, tile_test_loader, 'SliverNet/Test', device='cuda')
    print('Average IOU: {}, Accuracy:{}'.format(avg_iou, avg_accuracy))


if __name__ == '__main__':
    run()
