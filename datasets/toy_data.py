import torch
from torch.utils.data import Dataset


def gkern(kernlen=21, nsig=3):
    import numpy
    import scipy.stats as st

    """Returns a 2D Gaussian kernel array."""

    interval = (2 * nsig + 1.) / kernlen
    x = numpy.linspace(-nsig - interval / 2., nsig + interval / 2., kernlen + 1)
    kern1d = numpy.diff(st.norm.cdf(x))
    kernel_raw = numpy.sqrt(numpy.outer(kern1d, kern1d))
    kernel = kernel_raw / kernel_raw.sum()
    return torch.from_numpy(kernel)


class ToyData(Dataset):
    def __init__(self, N=512, device="cpu"):
        """N = number of subjects in a group"""

        kernel = gkern(kernlen=16, nsig=5).to(device)

        # Simulate Data
        torch.manual_seed(0)

        labels = torch.zeros((N * 2,), device=device)
        labels[N:] = 1

        # 2 confounding effects between 2 groups
        self.cf = torch.empty((N * 2,), device=device)
        self.cf[:N].uniform_(1, 4)
        self.cf[N:].uniform_(3, 6)

        # 2 major effects between 2 groups
        self.mf = torch.empty((N * 2,), device=device)
        self.mf[:N].uniform_(1, 4)
        self.mf[N:].uniform_(3, 6)

        # simulate images
        self.x = torch.zeros((N * 2, 1, 32, 32), device=device)
        self.y = torch.zeros((N * 2,), device=device)
        self.y[N:] = 1
        for i in range(N * 2):
            self.x[i, 0, :16, :16] = kernel * self.mf[i]
            self.x[i, 0, 16:, :16] = kernel * self.cf[i]
            self.x[i, 0, :16, 16:] = kernel * self.cf[i]
            self.x[i, 0, 16:, 16:] = kernel * self.mf[i]
            self.x[i] = self.x[i] + torch.normal(0, 0.01, size=(1, 32, 32), device=device)

    def __getitem__(self, index):
        return self.x[index], self.y[index], self.cf[index]

    def __len__(self):
        return len(self.y)

def full_toy_data(device="cpu"):
    num = 6
    all_samples = torch.empty((num, num, 32, 32), device=device)
    steps = torch.linspace(1, 6, num)
    kernel = gkern(kernlen=16, nsig=5)
    all_samples[:, :, :16, :16] = kernel * steps.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    all_samples[:, :, 16:, :16] = kernel * steps.unsqueeze(-1).unsqueeze(-1)
    all_samples[:, :, :16, 16:] = kernel * steps.unsqueeze(-1).unsqueeze(-1)
    all_samples[:, :, 16:, 16:] = kernel * steps.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    return all_samples
