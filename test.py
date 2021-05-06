import argparse
from typing import List
import torch
from eval import verification2
import logging
import os
import utils.backbones as backbones
from config import config as cfg


class CallBackVerification(object):
    def __init__(self, frequent, rank, val_targets, rec_prefix, image_size=(112, 112)):
        self.frequent: int = frequent
        self.rank: int = rank
        self.highest_acc: float = 0.0
        self.highest_acc_list: List[float] = [0.0] * len(val_targets)
        self.ver_list: List[object] = []
        self.ver_name_list: List[str] = []
        self.init_dataset(val_targets=val_targets, data_dir=rec_prefix, image_size=image_size)

    def ver_test(self, backbone: torch.nn.Module, global_step: int):
        results = []
        for i in range(len(self.ver_list)):
            acc1, std1, acc2, std2, xnorm, embeddings_list = verification2.test(
                self.ver_list[i], backbone, 10, 10)
            print('[%s][%d]XNorm: %f' % (self.ver_name_list[i], global_step, xnorm))
            print('[%s][%d]Accuracy-Flip: %1.5f+-%1.5f' % (self.ver_name_list[i], global_step, acc2, std2))
            if acc2 > self.highest_acc_list[i]:
                self.highest_acc_list[i] = acc2
            print(
                '[%s][%d]Accuracy-Highest: %1.5f' % (self.ver_name_list[i], global_step, self.highest_acc_list[i]))
            results.append(acc2)

    def init_dataset(self, val_targets, data_dir, image_size):
        for name in val_targets:
            path = os.path.join(data_dir, name + "_list.txt")
            if os.path.exists(path):
                data_set = verification2.load_data(path)
                self.ver_list.append(data_set)
                self.ver_name_list.append(name)
            else:
                print('no dir')

    def __call__(self, num_update, backbone: torch.nn.Module):
        backbone.eval()
        self.ver_test(backbone, num_update)
        backbone.train()


from collections import OrderedDict
def load_normal(load_path):
    state_dict = torch.load(load_path, map_location=torch.device('cpu'))
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if '.module.' in k:
            k = k.replace('.module.', '.')
        new_state_dict[k] = v
    return new_state_dict


def main(args):
    # net
    dropout = 0.4 if cfg.dataset is "webface" else 0
    backbone = backbones.__dict__[args.network](pretrained=False, dropout=dropout, fp16=cfg.fp16)
    state_dict = load_normal(args.resume)
    backbone.load_state_dict(state_dict)
    backbone = backbone.cuda()
    callback_verification = CallBackVerification(2000, 0, cfg.val_targets, cfg.rec)
    callback_verification(2000, backbone)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ArcFace Training')
    parser.add_argument('--network', type=str, default='iresnet18', help='backbone network')
    parser.add_argument('--resume', type=str, default=r'E:\pre-models\glint360k-iresnet18\backbone.pth', help='model resuming')
    args_ = parser.parse_args()
    main(args_)
