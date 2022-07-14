import glob

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms

from stylegan2.model import Generator

"""
WIP
"""

class FaceData(Dataset):
    def __init__(self, device, train=True):
        img_names = glob.glob("GS_data/cropped/*.png")
        img_names_sorted = sorted(img_names)
        N = len(img_names_sorted)

        self.data = torch.empty((N, 3, 224, 224))
        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(256),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

        imgs = []

        for imgfile in img_names_sorted:
            img = transform(Image.open(imgfile).convert("RGB"))
            imgs.append(img)

        imgs = torch.stack(imgs, 0).to(device)

        g_ema = Generator(256, 512, 8)
        g_ema.load_state_dict(torch.load(args.ckpt)["g_ema"], strict=False)
        g_ema.eval()
        g_ema = g_ema.to(device)

        self.y = torch.from_numpy(np.loadtxt('GS_data/gender.txt') > 0.5)
        self.cf = torch.from_numpy(np.loadtxt('GS_data/skincolor.txt'))

        if N != len(self.y) or N != len(self.cf):
            raise ValueError("Data of improper length")

        X_train, X_test, y_train, y_test, cf_train, cf_test = train_test_split(self.data, self.y, self.cf,
                                                                               stratify=self.y,
                                                                               test_size=0.25)

        if train:
            self.data = X_train.to(device=device)
            self.y = y_train.to(device=device, dtype=torch.float)
            self.cf = cf_train.to(device=device, dtype=torch.int)
        else:
            self.data = X_test.to(device=device)
            self.y = y_test.to(device=device, dtype=torch.float)
            self.cf = cf_test.to(device=device, dtype=torch.int)

    def __getitem__(self, index):
        return self.data[index], self.y[index], self.cf[index]

    def __len__(self):
        return len(self.y)
