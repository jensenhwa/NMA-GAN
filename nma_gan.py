import glob
import math

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
from PIL import Image
from matplotlib import pyplot as plt, gridspec
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms

from stylegan2.model import Generator

np.random.seed(0)
NOISE_DIM = 96


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


class ToyData(Dataset):
    def __init__(self, N=512):
        """N = number of subjects in a group"""

        def gkern(kernlen=21, nsig=3):
            import numpy
            import scipy.stats as st

            """Returns a 2D Gaussian kernel array."""

            interval = (2 * nsig + 1.) / (kernlen)
            x = numpy.linspace(-nsig - interval / 2., nsig + interval / 2., kernlen + 1)
            kern1d = numpy.diff(st.norm.cdf(x))
            kernel_raw = numpy.sqrt(numpy.outer(kern1d, kern1d))
            kernel = kernel_raw / kernel_raw.sum()
            return kernel

        ## Simulate Data
        torch.manual_seed(0)

        labels = torch.zeros((N * 2,))
        labels[N:] = 1

        # 2 confounding effects between 2 groups
        self.cf = torch.empty((N * 2,))
        self.cf[:N].uniform_(1, 4)
        self.cf[N:].uniform_(3, 6)

        # 2 major effects between 2 groups
        self.mf = torch.empty((N * 2,))
        self.mf[:N].uniform_(1, 4)
        self.mf[N:].uniform_(3, 6)

        # simulate images
        self.x = torch.zeros((N * 2, 1, 32, 32))
        self.y = torch.zeros((N * 2,))
        self.y[N:] = 1
        for i in range(N * 2):
            self.x[i, 0, :16, :16] = torch.from_numpy(gkern(kernlen=16, nsig=5)) * self.mf[i]
            self.x[i, 0, 16:, :16] = torch.from_numpy(gkern(kernlen=16, nsig=5)) * self.cf[i]
            self.x[i, 0, :16, 16:] = torch.from_numpy(gkern(kernlen=16, nsig=5)) * self.cf[i]
            self.x[i, 0, 16:, 16:] = torch.from_numpy(gkern(kernlen=16, nsig=5)) * self.mf[i]
            self.x[i] = self.x[i] + torch.normal(0, 0.01, size=(1, 32, 32))

    def __getitem__(self, index):
        return self.x[index], self.y[index], self.cf[index]

    def __len__(self):
        return len(self.y)


def sample_noise(batch_size, noise_dim, dtype=torch.float, device="cpu"):
    """
    Generate uniform random noise.

    Input:
    - batch_size: Integer giving the batch size of noise to generate.
    - noise_dim: Integer giving the dimension of noise to generate.

    Output:
    - A PyTorch Tensor of shape (batch_size, noise_dim) containing uniform
      random noise in the range (-1, 1).
    """
    noise = None
    noise = 2 * torch.rand((batch_size, noise_dim), dtype=dtype, device=device) - 1

    return noise


def discriminator1(latent_dim=100):
    model = nn.Sequential(
        nn.Linear(latent_dim + 1, 200),
        nn.LeakyReLU(),
        nn.Linear(200, 200),
        nn.LeakyReLU(),
        nn.Linear(200, 1),
    )
    return model


def discriminator2():
    model = nn.Sequential(
        nn.Linear(3, 6),
        nn.LeakyReLU(),
        nn.Linear(6, 6),
        nn.LeakyReLU(),
        nn.Linear(6, 1),
    )
    return model


def discriminator3():
    model = nn.Sequential(
        nn.Linear(2, 4),
        nn.LeakyReLU(),
        nn.Linear(4, 4),
        nn.LeakyReLU(),
        nn.Linear(4, 1),
    )
    return model


def generator(noise_dim=NOISE_DIM, latent_dim=100):
    model = nn.Sequential(
        nn.Linear(noise_dim, 1024),
        nn.ReLU(),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, latent_dim + 1),
        nn.Tanh()
    )
    return model


def encoder(classifier=None):
    """
    Build an encoder, optionally using the weights of a classifier.
    """
    model = nn.Sequential(
        nn.Conv2d(1, 2, 5, 2),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(2, 4, 3),
        nn.ReLU(),
        nn.Flatten(),
    )
    if classifier is not None:
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in classifier.state_dict().items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model


def decoder(encoder=None):
    model = nn.Sequential(
        nn.Linear(100, 200),
        nn.ReLU(),
        nn.Linear(200, 200),
        nn.ReLU(),
        nn.Linear(200, 32 * 32),
        nn.Sigmoid(),
        nn.Unflatten(1, (32, 32)),
    )
    return model


def classifier(noise_dim=NOISE_DIM):
    model = nn.Sequential(
        nn.Conv2d(1, 2, 5, 2),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(2, 4, 3),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(100, 1),
    )

    return model


def discriminator_loss(logits_real, logits_fake):
    """
    Computes the discriminator loss.

    Inputs:
    - logits_real: Tensor of shape (N,) giving scores for the real data.
    - logits_fake: Tensor of shape (N,) giving scores for the fake data.

    Returns:
    - loss: Tensor containing the scalar loss for the discriminator.
    """
    loss = 2 * F.binary_cross_entropy_with_logits(torch.cat((logits_real.squeeze(), logits_fake.squeeze())),
                                                  torch.tensor([1] * logits_real.shape[0] + [0] * logits_real.shape[0],
                                                               device=logits_real.device, dtype=torch.float))
    return loss


def generator_loss(logits_fake):
    """
    Computes the generator loss.

    Inputs:
    - logits_fake: Tensor of shape (N,) giving scores for the fake data.

    Returns:
    - loss: Tensor containing the scalar loss for the generator.
    """
    loss = F.binary_cross_entropy_with_logits(logits_fake, torch.ones_like(logits_fake))
    return loss


def get_optimizer(model, lr=1e-3, betas=(0.5, 0.999)):
    """
    Construct and return optimizer

    Input:
    - model: A PyTorch model that we want to optimize.

    Returns:
    - An Adam optimizer for the model with the desired hyperparameters.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=betas)
    return optimizer


def run_a_gan(loader_train, D2, D3, G, D_solver, G_solver, discriminator_loss, device, show_every=250,
              batch_size=128, num_epochs=10, l=0.001):
    """
    Train simple fair classification GAN
    """

    iter_count = 0
    for epoch in range(num_epochs):
        for x, y, cf, mf in loader_train:
            if len(x) != batch_size:
                continue

            for i in range(10):
                D_solver.zero_grad()
                real_data = x.view(-1, 1, 32, 32).to(device)
                preds = G(real_data).detach().squeeze()

                z_tilde = y
                y_prime = torch.rand((len(x),))
                s = cf  # sensitive atts

                logits_real2 = D2(torch.stack((s, y_prime, z_tilde), dim=1))
                logits_fake2 = D2(torch.stack((s, F.sigmoid(preds), z_tilde), dim=1))
                l2 = discriminator_loss(logits_real2, logits_fake2)

                logits_real3 = D3(torch.stack((y_prime, z_tilde), dim=1))
                logits_fake3 = D3(torch.stack((F.sigmoid(preds), z_tilde), dim=1))
                l3 = discriminator_loss(logits_real3, logits_fake3)

                d_total_error = - F.binary_cross_entropy_with_logits(preds, y) + l2 + l3
                d_total_error.backward()
                D_solver.step()
                # print(f"  d{i}: {d_total_error}")

            for i in range(10):
                G_solver.zero_grad()
                real_data = x.view(-1, 1, 32, 32).to(device)
                preds = G(real_data).squeeze()

                z_tilde = y
                y_prime = torch.rand((len(x),))
                s = cf  # sensitive atts

                logits_real2 = D2(torch.stack((s, y_prime, z_tilde), dim=1))
                logits_fake2 = D2(torch.stack((s, F.sigmoid(preds), z_tilde), dim=1))
                l2 = discriminator_loss(logits_real2, logits_fake2)

                logits_real3 = D3(torch.stack((y_prime, z_tilde), dim=1))
                logits_fake3 = D3(torch.stack((F.sigmoid(preds), z_tilde), dim=1))
                l3 = discriminator_loss(logits_real3, logits_fake3)

                l4 = (l2 - l3) ** 2
                g_error = l4 * l + F.binary_cross_entropy_with_logits(preds, y)
                g_error.backward()
                G_solver.step()
                # print(f"  g{i}: {g_error}")

            if (iter_count % show_every == 0):
                print('Iter: {}, D: {:.4}, G:{:.4}'.format(iter_count, d_total_error.item(), g_error.item()))
                print("loss: ", F.binary_cross_entropy_with_logits(preds, y).item())
                print()
            iter_count += 1

    return G, D2, D3


def run_real_gan(loader_train, D1, D2, D3, G, ENC, D_solver, G_solver, discriminator_loss, generator_loss, device,
                 save_filename, show_every=250,
                 batch_size=128, noise_size=96, num_epochs=10, l=0.001):
    """
    Train complete data generation GAN
    """
    try:
        iter_count = 0
        for epoch in range(num_epochs):
            for x, y, cf in loader_train:
                if len(x) != batch_size:
                    continue

                real_data = x
                for i in range(10):
                    D_solver.zero_grad()
                    features = ENC(real_data)
                    logits_real = D1(torch.cat((features, y.unsqueeze(1)), dim=1))

                    g_fake_seed = sample_noise(batch_size, noise_size, dtype=real_data.dtype, device=real_data.device)
                    fake_features_and_preds = G(g_fake_seed).detach()
                    fake_features_and_preds += 1
                    fake_features_and_preds /= 2
                    logits_fake = D1(fake_features_and_preds)

                    l1 = discriminator_loss(logits_real, logits_fake)
                    preds = fake_features_and_preds[:, -1]

                    z_tilde = y
                    y_prime = torch.rand((len(x),), dtype=torch.float, device=device)
                    s = cf  # sensitive atts

                    logits_real2 = D2(torch.stack((s, y_prime, z_tilde), dim=1))
                    logits_fake2 = D2(torch.stack((s, preds, z_tilde), dim=1))
                    l2 = discriminator_loss(logits_real2, logits_fake2)

                    logits_real3 = D3(torch.stack((y_prime, z_tilde), dim=1))
                    logits_fake3 = D3(torch.stack((preds, z_tilde), dim=1))
                    l3 = discriminator_loss(logits_real3, logits_fake3)

                    d_total_error = l1 + l2 + l3
                    d_total_error.backward()
                    D_solver.step()
                    # print(f"  d{i}: {d_total_error}")

                for i in range(10):
                    G_solver.zero_grad()
                    features = ENC(real_data)
                    logits_real = D1(torch.cat((features, y.unsqueeze(1)), dim=1))

                    g_fake_seed = sample_noise(batch_size, noise_size, dtype=real_data.dtype, device=real_data.device)
                    fake_features_and_preds = (G(g_fake_seed) + 1) / 2
                    logits_fake = D1(fake_features_and_preds)

                    l1 = discriminator_loss(logits_real, logits_fake)
                    preds = fake_features_and_preds[:, -1]

                    z_tilde = y
                    y_prime = torch.rand((len(x),), dtype=torch.float, device=device)
                    s = cf  # sensitive atts

                    logits_real2 = D2(torch.stack((s, y_prime, z_tilde), dim=1))
                    logits_fake2 = D2(torch.stack((s, preds, z_tilde), dim=1))
                    l2 = discriminator_loss(logits_real2, logits_fake2)

                    logits_real3 = D3(torch.stack((y_prime, z_tilde), dim=1))
                    logits_fake3 = D3(torch.stack((preds, z_tilde), dim=1))
                    l3 = discriminator_loss(logits_real3, logits_fake3)

                    l4 = (l2 - l3) ** 2
                    g_error = l4 * l + l1
                    g_error.backward()
                    G_solver.step()
                    # print(f"  g{i}: {g_error}")

                if (iter_count % show_every == 0):
                    print('Iter: {}, D: {:.4}, G:{:.4}'.format(iter_count, d_total_error.item(), g_error.item()))
                    # Need to compute classification accuracy, not generated images
                    print("loss: ", F.binary_cross_entropy_with_logits(preds, y).item())
                    print()
                iter_count += 1
    except KeyboardInterrupt:
        print("Interrupted")

    return G, ENC, D1, D2, D3


def run_face_gan(loader_train, D1, D2, D3, G, D_solver, G_solver, discriminator_loss, generator_loss, device,
                 save_filename, show_every=250,
                 batch_size=128, noise_size=96, num_epochs=10, l=0.001):
    """
    This does the same as run_real_gan but accepts features directly from loader_train, rather
    than using an independent encoder.
    """
    try:
        iter_count = 0
        for epoch in range(num_epochs):
            for features, y, cf in loader_train:
                if len(features) != batch_size:
                    continue

                for i in range(10):
                    D_solver.zero_grad()
                    logits_real = D1(torch.cat((features, y.unsqueeze(1)), dim=1))

                    g_fake_seed = sample_noise(batch_size, noise_size, dtype=features.dtype, device=features.device)
                    fake_features_and_preds = G(g_fake_seed).detach()
                    fake_features_and_preds += 1
                    fake_features_and_preds /= 2
                    logits_fake = D1(fake_features_and_preds)

                    l1 = discriminator_loss(logits_real, logits_fake)
                    preds = fake_features_and_preds[:, -1]

                    z_tilde = y
                    y_prime = torch.rand((len(features),), dtype=torch.float, device=device)
                    s = cf  # sensitive atts

                    logits_real2 = D2(torch.stack((s, y_prime, z_tilde), dim=1))
                    logits_fake2 = D2(torch.stack((s, preds, z_tilde), dim=1))
                    l2 = discriminator_loss(logits_real2, logits_fake2)

                    logits_real3 = D3(torch.stack((y_prime, z_tilde), dim=1))
                    logits_fake3 = D3(torch.stack((preds, z_tilde), dim=1))
                    l3 = discriminator_loss(logits_real3, logits_fake3)

                    d_total_error = l1 + l2 + l3
                    d_total_error.backward()
                    D_solver.step()
                    # print(f"  d{i}: {d_total_error}")

                for i in range(10):
                    G_solver.zero_grad()
                    logits_real = D1(torch.cat((features, y.unsqueeze(1)), dim=1))

                    g_fake_seed = sample_noise(batch_size, noise_size, dtype=features.dtype, device=features.device)
                    fake_features_and_preds = (G(g_fake_seed) + 1) / 2
                    logits_fake = D1(fake_features_and_preds)

                    l1 = discriminator_loss(logits_real, logits_fake)
                    preds = fake_features_and_preds[:, -1]

                    z_tilde = y
                    y_prime = torch.rand((len(features),), dtype=torch.float, device=device)
                    s = cf  # sensitive atts

                    logits_real2 = D2(torch.stack((s, y_prime, z_tilde), dim=1))
                    logits_fake2 = D2(torch.stack((s, preds, z_tilde), dim=1))
                    l2 = discriminator_loss(logits_real2, logits_fake2)

                    logits_real3 = D3(torch.stack((y_prime, z_tilde), dim=1))
                    logits_fake3 = D3(torch.stack((preds, z_tilde), dim=1))
                    l3 = discriminator_loss(logits_real3, logits_fake3)

                    l4 = (l2 - l3) ** 2
                    g_error = l4 * l + l1
                    g_error.backward()
                    G_solver.step()
                    # print(f"  g{i}: {g_error}")

                if (iter_count % show_every == 0):
                    print('Iter: {}, D: {:.4}, G:{:.4}'.format(iter_count, d_total_error.item(), g_error.item()))
                    # Need to compute classification accuracy, not generated images
                    print("loss: ", F.binary_cross_entropy_with_logits(preds, y).item())
                    print()
                iter_count += 1
            # if epoch == num_epochs - 1:
            #     show_images(imgs_numpy[0:16])
            #     plt.savefig(os.path.join(GOOGLE_DRIVE_PATH, save_filename))
    except KeyboardInterrupt:
        print("Interrupted")

    return G, D1, D2, D3


def show_images(images):
    images = torch.reshape(
        images, [images.shape[0], -1]
    )  # images reshape to (batch_size, D)
    sqrtn = int(math.ceil(math.sqrt(images.shape[0])))
    sqrtimg = int(math.ceil(math.sqrt(images.shape[1])))

    fig = plt.figure(figsize=(sqrtn, sqrtn))
    gs = gridspec.GridSpec(sqrtn, sqrtn)
    gs.update(wspace=0.05, hspace=0.05)

    for i, img in enumerate(images):
        ax = plt.subplot(gs[i])
        plt.axis("off")
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect("equal")
        plt.imshow(img.reshape([sqrtimg, sqrtimg]), cmap='gray')
        plt.colorbar()
    return


def train_decoder(loader_train, ENC, DEC, optim, device, batch_size=128, num_epochs=10, show_every=250):
    iter_count = 0
    for epoch in range(num_epochs):
        for x, y, cf, mf in loader_train:
            if len(x) != batch_size:
                continue
            optim.zero_grad()
            real_data = x.view(-1, 1, 32, 32).to(device)
            features = ENC(real_data).detach()

            x_hat = DEC(features)

            loss = F.binary_cross_entropy(x_hat, x.squeeze(), reduction="sum")

            loss.backward()
            optim.step()

            if (iter_count % show_every == 0):
                print('Iter: {}, loss: {}'.format(iter_count, loss.item()))
                imgs_numpy = x_hat.data.cpu()
                show_images(imgs_numpy[0:16])
                plt.show()
                print()
            iter_count += 1
        print('Iter: {}, loss: {}'.format(iter_count, loss.item()))
        imgs_numpy = x_hat.data.cpu()
        show_images(imgs_numpy[0:16])
        plt.show()
        print()
        plt.savefig("img.png")
    return DEC
