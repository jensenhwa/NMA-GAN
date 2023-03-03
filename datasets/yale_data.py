from pathlib import Path

import torch
from torchvision import transforms
from torch.utils.data import Dataset


class YaleData(Dataset):
    """
    Yale face data for use with diffusion autoencoders

    - self.x are the images
    - self.y is the pose (0 through 8 inclusive)
    - self.cf is the illumination (0 through 24 inclusive)
    """

    def __init__(self, set, device, dimension, cv_fold=None):
        if set == "train":
            self.x = torch.load(Path(__file__).parent / f"yale/X_train{('_f' + str(cv_fold)) if cv_fold else ''}.pt",
                                map_location=device)
            self.y = torch.load(Path(__file__).parent / f"yale/y_train{('_f' + str(cv_fold)) if cv_fold else ''}.pt",
                                map_location=device)
            self.cf = torch.load(Path(__file__).parent / f"yale/cf_train{('_f' + str(cv_fold)) if cv_fold else ''}.pt",
                                 map_location=device)
        elif set == "val":
            self.x = torch.load(Path(__file__).parent / f"yale/X_val{('_f' + str(cv_fold)) if cv_fold else ''}.pt",
                                map_location=device)
            self.y = torch.load(Path(__file__).parent / f"yale/y_val{('_f' + str(cv_fold)) if cv_fold else ''}.pt",
                                map_location=device)
            self.cf = torch.load(Path(__file__).parent / f"yale/cf_val{('_f' + str(cv_fold)) if cv_fold else ''}.pt",
                                 map_location=device)
        elif set == "test":
            self.x = torch.load(Path(__file__).parent / "yale/X_test.pt", map_location=device)
            self.y = torch.load(Path(__file__).parent / "yale/y_test.pt", map_location=device)
            self.cf = torch.load(Path(__file__).parent / "yale/cf_test.pt", map_location=device)
        else:
            raise NotImplementedError
        self.x = transforms.Resize(dimension)(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index], self.cf[index]

    def __len__(self):
        return len(self.y)
