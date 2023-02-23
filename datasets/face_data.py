from pathlib import Path

import torch
from torchvision import transforms
from torch.utils.data import Dataset


class FaceData(Dataset):
    """
    GS-PPB face data for use with diffusion autoencoders

    - self.x are the images
    - self.y is the gender (1 for female, 0 for male)
    - self.cf is the skin color (1 for lightest, 6 for darkest)
    """

    def __init__(self, set, device, dimension, cv_fold=None):
        # self.images = torch.load(Path(__file__).parent / "../diffae/images.pt")
        #
        # # self.cond = torch.load(Path(__file__).parent / "../diffae/cond_all.pt")
        # # self.xT = torch.load(Path(__file__).parent / "../diffae/xT_all.pt")
        # N = self.images.shape[0]
        #
        # self.y = torch.load(Path(__file__).parent / "../../GS_data/gender_diffae.pt")
        # self.cf = torch.load(Path(__file__).parent / "../../GS_data/skincolor_diffae.pt")
        #
        # if N != len(self.y) or N != len(self.cf):
        #     raise ValueError("Data of improper length")
        #
        # X_train, X_test, y_train, y_test, cf_train, cf_test = train_test_split(self.images, self.y, self.cf,
        #                                                                        stratify=self.y * 8 + self.cf,
        #                                                                        test_size=0.20)
        #
        # X_train, X_val, y_train, y_val, cf_train, cf_val = train_test_split(X_train, y_train, cf_train,
        #                                                                     stratify=y_train * 8 + cf_train,
        #                                                                     test_size=0.20)

        if set == "train":
            self.x = torch.load(Path(__file__).parent / f"face/balanced/X_train{('_f' + str(cv_fold)) if cv_fold else ''}.pt",
                                map_location=device)
            self.y = torch.load(Path(__file__).parent / f"face/balanced/y_train{('_f' + str(cv_fold)) if cv_fold else ''}.pt",
                                map_location=device)
            self.cf = torch.load(Path(__file__).parent / f"face/balanced/cf_train{('_f' + str(cv_fold)) if cv_fold else ''}.pt",
                                 map_location=device)
        elif set == "val":
            self.x = torch.load(Path(__file__).parent / f"face/balanced/X_val{('_f' + str(cv_fold)) if cv_fold else ''}.pt",
                                map_location=device)
            self.y = torch.load(Path(__file__).parent / f"face/balanced/y_val{('_f' + str(cv_fold)) if cv_fold else ''}.pt",
                                map_location=device)
            self.cf = torch.load(Path(__file__).parent / f"face/balanced/cf_val{('_f' + str(cv_fold)) if cv_fold else ''}.pt",
                                 map_location=device)
        elif set == "test":
            self.x = torch.load(Path(__file__).parent / "face/balanced/X_test.pt", map_location=device)
            self.y = torch.load(Path(__file__).parent / "face/balanced/y_test.pt", map_location=device)
            self.cf = torch.load(Path(__file__).parent / "face/balanced/cf_test.pt", map_location=device)
        else:
            raise NotImplementedError
        self.x = transforms.Resize(dimension)(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index], self.cf[index]

    def __len__(self):
        return len(self.y)


class FaceDataForLatent128(Dataset):
    """
    GS-PPB face data for training latent sampler of diffusion autoencoder
    """

    def __init__(self):
        self.images = torch.load(Path(__file__).parent / "../diffae/images.pt")
        self.images = transforms.Resize(128)(self.images)

    def __getitem__(self, index):
        return {'img': self.images[index], 'index': index}

    def __len__(self):
        return len(self.images)
