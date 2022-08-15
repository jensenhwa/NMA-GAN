from pathlib import Path

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


class FaceData(Dataset):
    """
    GS-PPB face data for use with diffusion autoencoders

    - self.x and self.x2 are the cond and xT tensors that make up the diffae's latent representation
    - self.y is the gender (1 for female, 0 for male)
    - self.cf is the skin color (1 for lightest, 6 for darkest)
    """

    def __init__(self, device, set):

        self.cond = torch.load(Path(__file__).parent / "../diffae/cond_all.pt")
        self.xT = torch.load(Path(__file__).parent / "../diffae/xT_all.pt")
        N = self.xT.shape[0]

        self.y = torch.load(Path(__file__).parent / "../../GS_data/gender_diffae.pt")
        self.cf = torch.load(Path(__file__).parent / "../../GS_data/skincolor_diffae.pt")

        if N != len(self.y) or N != len(self.cf):
            raise ValueError("Data of improper length")

        X_train, X_test, X2_train, X2_test, y_train, y_test, cf_train, cf_test = train_test_split(self.cond, self.xT,
                                                                                                  self.y, self.cf,
                                                                                                  stratify=self.y,
                                                                                                  test_size=0.20)

        X_train, X_val, X2_train, X2_val, y_train, y_val, cf_train, cf_val = train_test_split(X_train, X2_train,
                                                                                              y_train, cf_train,
                                                                                              stratify=y_train,
                                                                                              test_size=0.20)

        if set == "train":
            self.x = X_train.to(device=device)
            self.x2 = X2_train.to(device=device)
            self.y = y_train.to(device=device, dtype=torch.float)
            self.cf = cf_train.to(device=device, dtype=torch.int)
        elif set == "val":
            self.x = X_val.to(device=device)
            self.x2 = X2_val.to(device=device)
            self.y = y_val.to(device=device, dtype=torch.float)
            self.cf = cf_val.to(device=device, dtype=torch.int)
        elif set == "test":
            self.x = X_test.to(device=device)
            self.x2 = X2_test.to(device=device)
            self.y = y_test.to(device=device, dtype=torch.float)
            self.cf = cf_test.to(device=device, dtype=torch.int)
        else:
            raise NotImplementedError

    def __getitem__(self, index):
        return self.x[index], self.x2[index], self.y[index], self.cf[index]

    def __len__(self):
        return len(self.y)
