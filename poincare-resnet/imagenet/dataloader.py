from .path_config import data_dir
from .transforms import get_standard_transform

import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets

class TinyImageNetValDataset(Dataset):
    def __init__(self, root: str, class_to_idx: dict, transform=None):
        """
        root: tiny-imagenet-200/val
        class_to_idx: train셋에서 얻은 {wnid: idx} 맵
        """
        self.img_dir = os.path.join(root, 'images')
        ann_file = os.path.join(root, 'val_annotations.txt')
        # 파일명 → class id 매핑 읽기
        self.samples = []
        with open(ann_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                fname, wnid = parts[0], parts[1]
                if wnid not in class_to_idx:
                    continue
                label = class_to_idx[wnid]
                self.samples.append((fname, label))
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname, label = self.samples[idx]
        path = os.path.join(self.img_dir, fname)
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

class TinyImageNetDataLoaderFactory:
    def __init__(self, data_dir: str):
        """
        data_dir: tiny-imagenet-200이 위치한 상위 디렉터리 경로
        """
        base = os.path.join(data_dir, 'tiny-imagenet-200')
        self.train_dir = os.path.join(base, 'train')
        self.val_dir   = os.path.join(base, 'val')

        # transforms
        self.train_transform = get_standard_transform(train=True)
        self.val_transform   = get_standard_transform(train=False)

        # train set: ImageFolder를 사용하여 wnid→정수 라벨링 자동 생성
        self.train_set = datasets.ImageFolder(
            root=self.train_dir,
            transform=self.train_transform
        )
        # class_to_idx 사전 복사
        self.class_to_idx = self.train_set.class_to_idx

        # val set: custom Dataset
        self.val_set = TinyImageNetValDataset(
            root=self.val_dir,
            class_to_idx=self.class_to_idx,
            transform=self.val_transform
        )

    def create_train_loaders(self, batch_size: int, num_workers: int = 8):
        """
        train_loader, val_loader 반환
        """
        train_loader = DataLoader(
            dataset=self.train_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        val_loader = DataLoader(
            dataset=self.val_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        return train_loader, val_loader
