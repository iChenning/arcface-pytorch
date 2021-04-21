import os
from PIL import Image, ImageFile
from torchvision import transforms
from torch.utils.data import Dataset


def normal_trans(img_size):
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    test_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    return (train_transforms, test_transforms)


class MyDataset(Dataset):
    def __init__(self, txt_path, transform=None):
        assert os.path.exists(txt_path), "nonexistent:" + txt_path
        f = open(txt_path, 'r', encoding='utf-8')
        imgs = []
        labels = []
        for line in f:
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0].encode('utf-8'), int(words[1])))
            labels.append(int(words[1]))
        f.close()
        self.n_classes = self.labels_check(labels)
        self.imgs = imgs
        self.transform = transform

    def __getitem__(self, index):
        f_path, label = self.imgs[index]
        img = Image.open(f_path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        # return (img, label, f_path, index)
        return (img, label)

    def __len__(self):
        return len(self.imgs)

    def labels_check(self, labels):
        """
        判断labels是否是连续的，从0开始的，并返回类别数
        """
        labels_set = set(labels)
        labels_continuous = set(range(len(labels_set)))
        labels_diff = labels_continuous - labels_set
        assert len(labels_diff) == 0, print(labels_diff, len(labels))
        return len(labels_set)


