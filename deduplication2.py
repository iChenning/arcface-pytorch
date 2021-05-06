import argparse
import torch
import numpy as np
import utils.backbones as backbones
from utils.load_model import load_normal
from config import config as cfg
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
from thop import profile
import shutil
import os


test_trans = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])


test_trans2 = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])


def main(args):
    # net
    dropout = 0.4 if cfg.dataset is "webface" else 0
    backbone = backbones.__dict__[args.network](pretrained=False, dropout=dropout, fp16=cfg.fp16)
    state_dict = load_normal(args.resume)
    backbone.load_state_dict(state_dict)
    backbone = backbone.cuda()

    # macs-params
    macs, params = profile(backbone, inputs=(torch.rand(1, 3, 112, 112).cuda(),))
    print('macs:', macs, 'params:', params)

    # read path
    f_ = open(args.txt_dir, 'r')
    f_paths = []
    for line in f_.readlines():
        line = line.replace('\n', '')
        f_paths.append(line)
    f_.close()
    f_paths.sort()

    features1 = []
    features2 = []
    backbone.eval()
    for f_path in f_paths:
        img = Image.open(f_path).convert("RGB")
        w, h = img.size
        max_ = max(w, h)
        new_img = Image.new('RGB', (max_, max_), (0, 0, 0))
        new_img.paste(img, ((max_ - w) // 2, (max_ - h) // 2))

        img1 = test_trans(new_img)
        img1 = torch.unsqueeze(img1, 0)
        img2 = test_trans2(new_img)
        img2 = torch.unsqueeze(img2, 0)

        f1 = backbone(img1.cuda())
        f2 = backbone(img2.cuda())
        features1.append(f1.cpu().data)
        features2.append(f2.cpu().data)
    features1 = torch.cat(features1, 0)
    features2 = torch.cat(features2, 0)
    features = features1 + features2
    features = F.normalize(features)
    s = torch.mm(features, features.T)
    s = s.cpu().data.numpy()
    s[list(range(s.shape[0])), list(range(s.shape[0]))] = 0
    s_argmax = s.argmax(axis=1)
    s_max = s.max(axis=1)
    s_new = np.concatenate((s_max[np.newaxis, :], s_argmax[np.newaxis, :]), axis=0)
    r_ = r'E:\depu\glint360k-se_iresnet100-2'
    r_2 = r'C:\Users\amlogic\datasets\san_1920'
    for i in range(s.shape[0]):
        if s_max[i] > 0.55:
            if not os.path.exists(os.path.join(r_, str(i).zfill(3) + '_' + str(int(s_max[i] * 100)))):
                os.makedirs(os.path.join(r_, str(i).zfill(3) + '_' + str(int(s_max[i] * 100))))
            old_p = f_paths[i]
            name_ = os.path.split(old_p)[-1]
            new_p = os.path.join(r_, str(i).zfill(3) + '_' + str(int(s_max[i] * 100)), name_)
            shutil.copy(old_p, new_p)
            id_fre = name_.split('_')[-1]
            old_p = os.path.join(r_2, id_fre)
            new_p = os.path.join(r_, str(i).zfill(3) + '_' + str(int(s_max[i] * 100)), id_fre)
            shutil.copy(old_p, new_p)

            old_p = f_paths[s_argmax[i]]
            name_ = os.path.split(old_p)[-1]
            new_p = os.path.join(r_, str(i).zfill(3) + '_' + str(int(s_max[i] * 100)), name_)
            shutil.copy(old_p, new_p)
            id_fre = name_.split('_')[-1]
            old_p = os.path.join(r_2, id_fre)
            new_p = os.path.join(r_, str(i).zfill(3) + '_' + str(int(s_max[i] * 100)), id_fre)
            shutil.copy(old_p, new_p)

    print('done')



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ArcFace Training')
    parser.add_argument('--network', type=str, default='se_iresnet100', help='backbone network')
    parser.add_argument('--resume', type=str, default=r'E:\pre-models\glint360k-se_iresnet100\backbone.pth')
    parser.add_argument('--txt_dir', type=str, default='data_list/san_results-single-alig.txt')
    parser.add_argument('--save_root', type=str, default=r'E:\depu')
    parser.add_argument('--san_1920_dir', type=str, default=r'E:\datasets\san_1920')
    args_ = parser.parse_args()
    main(args_)
