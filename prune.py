import argparse
import torch
import torch.nn as nn
import utils.backbones as backbones
from utils.load_model import load_normal
from config import config as cfg
from thop import profile



def main(args):
    # net
    backbone = backbones.__dict__[args.network](pretrained=False, dropout=0.0, fp16=False)
    state_dict = load_normal(args.resume)
    backbone.load_state_dict(state_dict)
    backbone = backbone.cuda()
    print(backbone)

    #
    n_layers = ['layer1', 'layer2', 'layer3', 'layer4']
    n_bns = ['bn1']

    # pruned sta
    total = 0
    bn = []
    for n_layer in n_layers:
        layer = getattr(backbone, n_layer)
        for d in range(len(layer)):
            block = layer[d]
            for n_bn in n_bns:
                m = getattr(block, n_bn)
                if isinstance(m, nn.BatchNorm2d):
                    total += m.weight.data.shape[0]
                    bn.extend(m.weight.data.abs().clone())
    # m = getattr(backbone, 'bn2')
    # total += m.weight.data.shape[0]
    # bn.extend(m.weight.data.abs().clone())
    bn = torch.tensor(bn)

    y, i = torch.sort(bn)
    thre_index = int(total * args.percent)
    thre = y[thre_index]

    pruned = 0
    cfg = []
    cfg_mask = []
    for n_layer in n_layers:
        layer = getattr(backbone, n_layer)
        for d in range(len(layer)):
            block = layer[d]
            for n_bn in n_bns:
                m = getattr(block, n_bn)
                weight_copy = m.weight.data.abs().clone()
                mask = weight_copy.gt(thre).float().cuda()
                pruned = pruned + mask.shape[0] - torch.sum(mask)
                m.weight.data.mul_(mask)
                m.bias.data.mul_(mask)
                cfg.append(int(torch.sum(mask)))
                cfg_mask.append(mask.clone())
    # m = getattr(backbone, 'bn2')
    # weight_copy = m.weight.data.abs().clone()
    # mask = weight_copy.gt(thre).float().cuda()
    # pruned = pruned + mask.shape[0] - torch.sum(mask)
    # m.weight.data.mul_(mask)
    # m.bias.data.mul_(mask)
    # cfg.append(int(torch.sum(mask)))
    # cfg_mask.append(mask.clone())

    print('pruned_ratio =', pruned / total)

    print('Pre-processing Successful!')

    print(len(cfg), cfg)
    f_ = open(r'E:\pruned_info\glint360k-se_iresnet100.txt', 'w')
    for c in cfg:
        f_.write(str(c) + ' ')
    f_.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ArcFace Training')

    parser.add_argument('--network', type=str, default='se_iresnet100', help='backbone network')
    parser.add_argument('--resume', type=str, default=r'E:\pre-models\glint360k-se_iresnet100\backbone.pth')
    parser.add_argument('--percent', type=float, default=0.3,
                        help='scale sparse rate (default: 0.5)')

    args_ = parser.parse_args()
    main(args_)
