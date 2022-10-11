import math
from multiprocessing import get_context

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
from matplotlib import pyplot as plt, gridspec
from pytorch_lightning import LightningModule
from torch import nn, distributed
from torch.utils.data import DataLoader, DistributedSampler

from diffae.choices import TrainMode
from datasets.face_data import FaceData

np.random.seed(0)
NOISE_DIM = 96


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


def discriminator2v(v_len=100):
    model = nn.Sequential(
        nn.Linear(1 + v_len + 1, 200),
        nn.ReLU(),
        nn.Linear(200, 100),
        nn.ReLU(),
        nn.Linear(100, 1),
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


def discriminator3v(v_len=100):
    model = nn.Sequential(
        nn.Linear(v_len + 1, 200),
        nn.ReLU(),
        nn.Linear(200, 100),
        nn.ReLU(),
        nn.Linear(100, 1),
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


def encoder(classifier=None, final_activation="relu"):
    """
    Build an encoder, optionally using the weights of a classifier.
    """
    if final_activation == "relu":
        model = nn.Sequential(
            nn.Conv2d(1, 2, 5, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(2, 4, 3),
            nn.ReLU(),
            nn.Flatten(),
        )
    elif final_activation == "none":
        model = nn.Sequential(
            nn.Conv2d(1, 2, 5, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(2, 4, 3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(100, 10)
        )
    else:
        raise ValueError("Invalid argument for final_activation")

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


def get_optimizer(model, lr=1e-4):
    """
    Construct and return optimizer

    Input:
    - model: A PyTorch model that we want to optimize.

    Returns:
    - An Adam optimizer for the model with the desired hyperparameters.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    return optimizer


def run_regularized_classifier(loader_train, G, G_solver, device, show_every=250,
              batch_size=128, num_epochs=10, l=0.001):
    """
    Train simple fair classification GAN
    """

    iter_count = 0
    for epoch in range(num_epochs):
        for x, y, cf in loader_train:
            if len(x) != batch_size:
                continue

            for i in range(10):
                G_solver.zero_grad()
                real_data = x.view(-1, 1, 32, 32).to(device)
                preds = G(real_data).squeeze()

                eo_yzero = torch.abs(F.relu(preds[(y == 0) & (cf >= 3)]).sum() - F.relu(preds[(y == 0) & (cf < 3)]).sum())
                eo_yone = torch.abs(F.relu(preds[(y == 1) & (cf < 4)]).sum() - F.relu(preds[(y == 1) & (cf >= 4)]).sum())

                g_error = (eo_yzero + eo_yone) * l + F.binary_cross_entropy_with_logits(preds, y)
                g_error.backward()
                G_solver.step()
                # print(f"  g{i}: {g_error}")

            if (iter_count % show_every == 0):
                print('Iter: {}, G:{:.4}'.format(iter_count, g_error.item()))
                print("loss: ", F.binary_cross_entropy_with_logits(preds, y).item())
                print()
            iter_count += 1

    return G


def run_a_gan(loader_train, D2, D3, G, D_solver, G_solver, discriminator_loss, device, show_every=250,
              batch_size=128, num_epochs=10, l=0.001):
    """
    Train simple fair classification GAN
    """

    iter_count = 0
    for epoch in range(num_epochs):
        for x, y, cf in loader_train:
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


def run_v_gan(loader_train, D2, D3, ENC, FF, D_solver, G_solver, discriminator_loss, device, show_every=250,
              batch_size=128, num_epochs=10, l=1, acc_data=None):
    """
    Train simple fair classification GAN in v-space
    """

    iter_count = 0
    accs = []
    for epoch in range(num_epochs):
        for x, y, cf in loader_train:
            if len(x) != batch_size:
                continue

            for i in range(10):
                D_solver.zero_grad()
                real_data = x.view(-1, 1, 32, 32).to(device)
                features = ENC(real_data)
                preds = FF(features).detach().squeeze()

                z_tilde = y.unsqueeze(1)
                v_prime = features.detach().clone()
                v_prime[y == 0] = v_prime[y == 0][torch.randint(v_prime[y == 0].size()[0], (v_prime[y == 0].size()[0],))]
                v_prime[y == 1] = v_prime[y == 1][torch.randint(v_prime[y == 1].size()[0], (v_prime[y == 1].size()[0],))]
#                 v_prime = features.detach()[torch.randperm(features.size()[0])]
                s = cf.unsqueeze(1)  # sensitive atts
                # print(v_prime)
                # print(F.sigmoid(features))
                # if i == 9:
                #     assert False

                logits_real2 = D2(torch.cat((s, v_prime, z_tilde), dim=1))
                logits_fake2 = D2(torch.cat((s, features, z_tilde), dim=1))
                l2 = discriminator_loss(logits_real2, logits_fake2)

                logits_real3 = D3(torch.cat((v_prime, z_tilde), dim=1))
                logits_fake3 = D3(torch.cat((features, z_tilde), dim=1))
                l3 = discriminator_loss(logits_real3, logits_fake3)

                d_total_error = - F.binary_cross_entropy_with_logits(preds, y) + l2 + l3
                # print(- F.binary_cross_entropy_with_logits(preds, y))
                # print(l2)
                # print(l3)
                d_total_error.backward()
                D_solver.step()
                # print(f"  d{i}: {d_total_error}")

            for i in range(10):
                G_solver.zero_grad()
                real_data = x.view(-1, 1, 32, 32).to(device)
                features = ENC(real_data)
                preds = FF(features).squeeze()

                z_tilde = y.unsqueeze(1)
                v_prime = features.detach().clone()
                v_prime[y == 0] = v_prime[y == 0][torch.randint(v_prime[y == 0].size()[0], (v_prime[y == 0].size()[0],))]
                v_prime[y == 1] = v_prime[y == 1][torch.randint(v_prime[y == 1].size()[0], (v_prime[y == 1].size()[0],))]
#                 v_prime = features.detach()[torch.randperm(features.size()[0])]
                s = cf.unsqueeze(1)  # sensitive atts

                logits_real2 = D2(torch.cat((s, v_prime, z_tilde), dim=1))
                logits_fake2 = D2(torch.cat((s, features, z_tilde), dim=1))
                l2 = discriminator_loss(logits_real2, logits_fake2)

                logits_real3 = D3(torch.cat((v_prime, z_tilde), dim=1))
                logits_fake3 = D3(torch.cat((features, z_tilde), dim=1))
                l3 = discriminator_loss(logits_real3, logits_fake3)

                l4 = (l2 - l3) ** 2
                g_error = l4 * l + F.binary_cross_entropy_with_logits(preds, y)
                g_error.backward()
                G_solver.step()
                # print(f"  g{i}: {g_error}")

            # print(v_prime)
            # print(features)
            # print(F.sigmoid(features))
            with torch.no_grad():
                if acc_data is not None:
                    features = ENC(acc_data.x.view(-1, 1, 32, 32)).squeeze()
                    y_preds = FF(features).squeeze()
                    y_preds = (torch.sign(y_preds) + 1) / 2
                    train_acc = (torch.sum(y_preds == acc_data.y) / len(acc_data)).item()
                    print("iter =", iter_count, ", acc =", train_acc)
                    accs.append(train_acc)
                    plt.close()
                    plt.plot(range(iter_count+1), accs)
                    plt.savefig("acc_per_iter.png")

                if (iter_count % show_every == 0):
                    print('Iter: {}, D: {:.4}, G:{:.4}'.format(iter_count, d_total_error.item(), g_error.item()))
                    print("loss: ", F.binary_cross_entropy_with_logits(preds, y).item())
                    print()
            iter_count += 1

    return D2, D3, ENC, FF


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


def sample_face_noise(model, batch_size, device):

    T = 20
    T_latent = 200
    N = batch_size
    sampler = model.conf._make_diffusion_conf(T).make_sampler()
    latent_sampler = model.conf._make_latent_diffusion_conf(T_latent).make_sampler()
    x_T = torch.randn(N,
                      3,
                      model.conf.img_size,
                      model.conf.img_size,
                      device=device)
    latent_noise = torch.randn(len(x_T), model.conf.style_ch, device=device)

    cond = latent_sampler.sample(
        model=model.ema_model.latent_net,
        noise=latent_noise,
        clip_denoised=model.conf.latent_clip_sample,
    )
    cond = cond * model.conds_std.to(device) + model.conds_mean.to(device)
    return cond


def ema(source, target, decay):
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key].data.copy_(target_dict[key].data * decay +
                                    source_dict[key].data * (1 - decay))


class FaceGAN(LightningModule):
    def __init__(self, encoder, v_len: int, l: float, batch_size: int, **kwargs):
        super().__init__()
        self.l = l
        assert batch_size % get_world_size() == 0
        self.batch_size = batch_size // get_world_size()
        self.D2 = discriminator2v(v_len=v_len)
        self.D3 = discriminator3v(v_len=v_len)
        self.encoder = encoder
        self.FF = nn.Sequential(
            nn.Linear(512, 1),
        )

    def setup(self, stage=None) -> None:
        """
        make datasets & seeding each worker separately
        """
        ##############################################
        # NEED TO SET THE SEED SEPARATELY HERE
        if self.encoder.conf.seed is not None:
            seed = self.encoder.conf.seed * get_world_size() + self.global_rank
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            print('local seed:', seed)
        ##############################################

        self.train_data = FaceData(set="train", device=self.device, dimension=128)
        print('train data:', len(self.train_data))
        self.val_data = FaceData(set="val", device=self.device, dimension=128)
        print('val data:', len(self.val_data))
        # self.test_data = FaceData(set="test", device=self.device)
        # print('test data:', len(self.test_data))

    def train_dataloader(self):
        print('on train dataloader start ...')
        sampler = DistributedSampler(self.train_data, shuffle=True, drop_last=True)
        return DataLoader(self.train_data, batch_size=self.batch_size, sampler=sampler, shuffle=False,
                          num_workers=2,
                          pin_memory=True,
                          drop_last=True,
                          multiprocessing_context=get_context('fork'),)

    def val_dataloader(self):
        print('on val dataloader start ...')
        sampler = DistributedSampler(self.val_data, shuffle=False, drop_last=False)
        return DataLoader(self.val_data, batch_size=min(len(self.val_data) // get_world_size(), self.batch_size),
                          sampler=sampler,
                          shuffle=False,
                          num_workers=2,
                          pin_memory=True,
                          drop_last=True,
                          multiprocessing_context=get_context('fork'),)

    # def test_dataloader(self):
    #     print('on test dataloader start ...')
    #     print(self.batch_size)
    #     sampler = DistributedSampler(self.test_data, shuffle=False, drop_last=False)
    #     return DataLoader(self.test_data, batch_size=min(len(self.test_data) // get_world_size(), self.batch_size),
    #                       sampler=sampler,
    #                       shuffle=False,
    #                       num_workers=2,
    #                       pin_memory=True,
    #                       drop_last=True,
    #                       multiprocessing_context=get_context('fork'), )

    def training_step(self, batch, batch_idx, optimizer_idx):
        print("train step")
        # for p in self.encoder.model.input_blocks[0][0].parameters():
        #     print(p.requires_grad)
        x, y, cf = batch

        # train discriminator
        if optimizer_idx == 0:
            print("dis")
            with torch.no_grad():
                features = self.encoder.encode(x)
                preds = self.FF(features).detach().squeeze()

            z_tilde = y.unsqueeze(1)

            v_prime = features.detach().clone()
            v_prime[y == 0] = v_prime[y == 0][torch.randint(v_prime[y == 0].size()[0], (v_prime[y == 0].size()[0],))]
            v_prime[y == 1] = v_prime[y == 1][torch.randint(v_prime[y == 1].size()[0], (v_prime[y == 1].size()[0],))]

            s = cf.unsqueeze(1)  # sensitive atts

            logits_real2 = self.D2(torch.cat((s, v_prime, z_tilde), dim=1))
            logits_fake2 = self.D2(torch.cat((s, features, z_tilde), dim=1))
            l2 = discriminator_loss(logits_real2, logits_fake2)

            logits_real3 = self.D3(torch.cat((v_prime, z_tilde), dim=1))
            logits_fake3 = self.D3(torch.cat((features, z_tilde), dim=1))
            l3 = discriminator_loss(logits_real3, logits_fake3)

            # Technically, BCE loss term is unnecessary here
            loss = - F.binary_cross_entropy_with_logits(preds, y) + l2 + l3
            print(loss)
            self.log("d_loss", loss, prog_bar=True)
            return loss

        # train generator
        if optimizer_idx == 1:
            print("gen")
            features = self.encoder.encode(x)
            preds = self.FF(features).squeeze()
            z_tilde = y.unsqueeze(1)

            v_prime = features.detach().clone()
            v_prime[y == 0] = v_prime[y == 0][torch.randint(v_prime[y == 0].size()[0], (v_prime[y == 0].size()[0],))]
            v_prime[y == 1] = v_prime[y == 1][torch.randint(v_prime[y == 1].size()[0], (v_prime[y == 1].size()[0],))]

            s = cf.unsqueeze(1)

            logits_real2 = self.D2(torch.cat((s, v_prime, z_tilde), dim=1))
            logits_fake2 = self.D2(torch.cat((s, features, z_tilde), dim=1))
            l2 = discriminator_loss(logits_real2, logits_fake2)

            logits_real3 = self.D3(torch.cat((v_prime, z_tilde), dim=1))
            logits_fake3 = self.D3(torch.cat((features, z_tilde), dim=1))
            l3 = discriminator_loss(logits_real3, logits_fake3)

            l4 = (l2 - l3) ** 2

            t, weight = self.encoder.T_sampler.sample(len(x), x.device)
            losses = self.encoder.sampler.training_losses(model=self.encoder.model,
                                                  x_start=x,
                                                  t=t)
            diffae_loss = losses['loss'].mean()
            # divide by accum batches to make the accumulated gradient exact!
            for key in ['loss', 'vae', 'latent', 'mmd', 'chamfer', 'arg_cnt']:
                if key in losses:
                    losses[key] = self.all_gather(losses[key]).mean()

            if self.global_rank == 0:
                self.logger.experiment.add_scalar('loss', losses['loss'],
                                                  self.encoder.num_samples)
                for key in ['vae', 'latent', 'mmd', 'chamfer', 'arg_cnt']:
                    if key in losses:
                        self.logger.experiment.add_scalar(
                            f'loss/{key}', losses[key], self.encoder.num_samples)

            loss = l4 * self.l + F.binary_cross_entropy_with_logits(preds, y) + diffae_loss
            print(loss)
            self.log("g_loss", loss, prog_bar=True)
            return loss

    def on_train_batch_end(self, outputs, batch, batch_idx: int) -> None:
        if self.encoder.is_last_accum(batch_idx):
            # only apply ema on the last gradient accumulation step,
            # if it is the iteration that has optimizer.step()
            if self.encoder.conf.train_mode == TrainMode.latent_diffusion:
                # it trains only the latent hence change only the latent
                ema(self.encoder.model.latent_net, self.encoder.ema_model.latent_net,
                    self.encoder.conf.ema_decay)
            else:
                ema(self.encoder.model, self.encoder.ema_model, self.encoder.conf.ema_decay)

    def on_before_optimizer_step(self, optimizer: torch.optim.Optimizer,
                                 optimizer_idx: int) -> None:
        self.encoder.on_before_optimizer_step(optimizer, optimizer_idx)

    def validation_step(self, batch, batch_idx):
        print("val step")
        for p in self.encoder.model.input_blocks[0][0].parameters():
            print(p.requires_grad)
        x, y, cf = batch
        features = self.encoder.encode(x)
        y_preds = self.FF(features).squeeze()
        y_preds = (torch.sign(y_preds) + 1) / 2
        # Classes range from 1 to 6 inclusive (no 0)
        one_hot = torch.nn.functional.one_hot(cf.long(), num_classes=7)[:, 1:].bool()
        # accuracies by skin class
        train_accs = torch.sum(one_hot & (
                    y_preds.unsqueeze(1).expand(one_hot.shape) * one_hot == y.unsqueeze(1).expand(
                one_hot.shape) * one_hot), dim=0) / torch.sum(one_hot, dim=0)
        train_acc = (torch.sum(y_preds == y) / len(x)).item()
        print(train_accs)
        self.log("per_skin_acc", {str(i): x for i, x in enumerate(train_accs)})
        self.log("acc", train_acc)

    def configure_optimizers(self):
        D_solver = get_optimizer(nn.ModuleList([self.D2, self.D3]))
        G_solver = get_optimizer(nn.ModuleList([self.FF, self.encoder]))
        return D_solver, G_solver


def get_world_size():
    if distributed.is_initialized():
        return distributed.get_world_size()
    else:
        return 1


def run_face_gan(loader_train, model, D2, D3, FF, D_solver, G_solver, discriminator_loss, show_every=250,
                 batch_size=128, num_epochs=10, l=0.001, acc_data=None):
    """
    This does the same as run_real_gan but accepts features directly from loader_train, rather
    than using an independent encoder.
    """
    try:
        iter_count = 0
        accs = []
        for epoch in range(num_epochs):
            for features, _, y, cf in loader_train:
                if len(features) != batch_size:
                    continue

                for i in range(10):
                    D_solver.zero_grad()
                    preds = FF(features).detach().squeeze()

                    z_tilde = y.unsqueeze(1)
                    v_prime = sample_face_noise(model, batch_size, features.device)
                    s = cf.unsqueeze(1)  # sensitive atts
                    # print(v_prime)
                    # print(F.sigmoid(features))
                    # if i == 9:
                    #     assert False

                    logits_real2 = D2(torch.cat((s, v_prime, z_tilde), dim=1))
                    logits_fake2 = D2(torch.cat((s, features, z_tilde), dim=1))
                    l2 = discriminator_loss(logits_real2, logits_fake2)

                    logits_real3 = D3(torch.cat((v_prime, z_tilde), dim=1))
                    logits_fake3 = D3(torch.cat((features, z_tilde), dim=1))
                    l3 = discriminator_loss(logits_real3, logits_fake3)

                    d_total_error = - F.binary_cross_entropy_with_logits(preds, y) + l2 + l3
                    # print(- F.binary_cross_entropy_with_logits(preds, y))
                    # print(l2)
                    # print(l3)
                    d_total_error.backward()
                    D_solver.step()
                    # print(f"  d{i}: {d_total_error}")

                for i in range(10):
                    G_solver.zero_grad()
                    preds = FF(features).squeeze()
                    z_tilde = y.unsqueeze(1)
                    v_prime = sample_face_noise(model, batch_size, features.device)
                    s = cf.unsqueeze(1)

                    logits_real2 = D2(torch.cat((s, v_prime, z_tilde), dim=1))
                    logits_fake2 = D2(torch.cat((s, features, z_tilde), dim=1))
                    l2 = discriminator_loss(logits_real2, logits_fake2)

                    logits_real3 = D3(torch.cat((v_prime, z_tilde), dim=1))
                    logits_fake3 = D3(torch.cat((features, z_tilde), dim=1))
                    l3 = discriminator_loss(logits_real3, logits_fake3)

                    l4 = (l2 - l3) ** 2
                    g_error = l4 * l + F.binary_cross_entropy_with_logits(preds, y)
                    g_error.backward()
                    G_solver.step()

                # print(v_prime)
                # print(features)
                # print(F.sigmoid(features))
                with torch.no_grad():
                    if acc_data is not None:
                        y_preds: torch.Tensor = FF(acc_data.x).squeeze()
                        y_preds = (torch.sign(y_preds) + 1) / 2
                        # Classes range from 1 to 6 inclusive (no 0)
                        one_hot = torch.nn.functional.one_hot(acc_data.cf.long())[:, 1:].bool()
                        # accuracies by skin class
                        train_accs = torch.sum(one_hot & (y_preds.unsqueeze(1).expand(one_hot.shape) * one_hot == acc_data.y.unsqueeze(1).expand(one_hot.shape) * one_hot), dim=0) / torch.sum(one_hot, dim=0)
                        print(train_accs)
                        train_acc = (torch.sum(y_preds == acc_data.y) / len(acc_data)).item()
                        print("iter =", iter_count, ", acc =", train_acc)
                        accs.append(train_acc)
                        plt.close()
                        plt.plot(range(iter_count + 1), accs)
                        plt.savefig("face_acc_per_iter.png")

                    if (iter_count % show_every == 0):
                        print('Iter: {}, D: {:.4}, G:{:.4}'.format(iter_count, d_total_error.item(), g_error.item()))
                        print("loss: ", F.binary_cross_entropy_with_logits(preds, y).item())
                        print()
                iter_count += 1

    except KeyboardInterrupt:
        print("Interrupted")

    return D2, D3, FF


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
