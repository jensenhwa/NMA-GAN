import math
import pickle

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
from matplotlib import pyplot as plt, gridspec
from torch import nn

from synthetic_experiments import get_dcorr

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
                                                  torch.cat((torch.ones(logits_real.shape[0], device=logits_real.device),
                                                             torch.zeros(logits_fake.shape[0], device=logits_fake.device))))
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
              batch_size=128, num_epochs=10, l=0.001, acc_data=None, name="reg_temp"):
    """
    Train simple fair classification GAN
    """

    iter_count = 0
    accs = []
    dcors = []
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

            with torch.no_grad():
                if acc_data is not None:
                    y_preds_logits = G(acc_data.x.view(-1, 1, 32, 32)).squeeze()
                    y_preds = (torch.sign(y_preds_logits) + 1) / 2
                    train_acc = (torch.sum(y_preds == acc_data.y) / len(acc_data)).item()
                    dcorr = (get_dcorr(y_preds_logits[acc_data.y == 0], acc_data.cf[acc_data.y == 0])
                             + get_dcorr(y_preds_logits[acc_data.y == 1], acc_data.cf[acc_data.y == 1])) / 2
                    accs.append(train_acc)
                    dcors.append(dcorr)
                    plt.close()
                    plt.plot(range(iter_count+1), accs)
                    plt.plot(range(iter_count+1), dcors)

            if (iter_count % show_every == 0):
                if acc_data is not None:
                    print("iter =", iter_count, ", acc =", train_acc, ", dcorr =", dcorr)
                print('Iter: {}, G:{:.4}'.format(iter_count, g_error.item()))
                print("loss: ", F.binary_cross_entropy_with_logits(preds, y).item())
                print()
                plt.show()
            iter_count += 1

    with open(f"{name}_dcors.pkl", "wb") as fp:
        pickle.dump(dcors, fp)
    with open(f"{name}_accs.pkl", "wb") as fp:
        pickle.dump(accs, fp)
    return G


def run_a_gan(loader_train, D2, D3, G, D_solver, G_solver, discriminator_loss, device, show_every=250,
              batch_size=128, num_epochs=10, l=0.001, acc_data=None, name="y_temp"):
    """
    Train simple fair classification GAN
    """

    iter_count = 0
    accs = []
    dcors = []
    for epoch in range(num_epochs):
        for x, y, cf in loader_train:
            if len(x) != batch_size:
                continue

            for i in range(10):
                D_solver.zero_grad()
                real_data = x.view(-1, 1, 32, 32).to(device)
                preds = G(real_data).detach().squeeze()

                z_tilde = y
                y_prime = torch.rand((len(x),), device=device) * 0.5
                y_prime[y == 1] = y_prime[y == 1] + 0.5
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
                y_prime = torch.rand((len(x),), device=device)
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


            with torch.no_grad():
                if acc_data is not None:
                    y_preds_logits = G(acc_data.x.view(-1, 1, 32, 32)).squeeze()
                    y_preds = (torch.sign(y_preds_logits) + 1) / 2
                    train_acc = (torch.sum(y_preds == acc_data.y) / len(acc_data)).item()
                    dcorr = (get_dcorr(y_preds_logits[acc_data.y == 0], acc_data.cf[acc_data.y == 0])
                             + get_dcorr(y_preds_logits[acc_data.y == 1], acc_data.cf[acc_data.y == 1])) / 2
                    accs.append(train_acc)
                    dcors.append(dcorr)
                    plt.close()
                    plt.plot(range(iter_count+1), accs)
                    plt.plot(range(iter_count+1), dcors)
                    # plt.savefig("acc_per_iter.png")

                if (iter_count % show_every == 0):
                    if acc_data is not None:
                        print("iter =", iter_count, ", acc =", train_acc, ", dcorr =", dcorr)
                    print('Iter: {}, D: {:.4}, G:{:.4}'.format(iter_count, d_total_error.item(), g_error.item()))
                    print("loss: ", F.binary_cross_entropy_with_logits(preds, y).item())
                    print()
                    plt.show()
            iter_count += 1

    with open(f"{name}_dcors.pkl", "wb") as fp:
        pickle.dump(dcors, fp)
    with open(f"{name}_accs.pkl", "wb") as fp:
        pickle.dump(accs, fp)
    return G, D2, D3


def run_v_gan(loader_train, D2, D3, ENC, FF, D_solver, G_solver, discriminator_loss, device, show_every=250,
              batch_size=128, num_epochs=10, l=1, acc_data=None, name="temp"):
    """
    Train simple fair classification GAN in v-space
    """

    iter_count = 0
    accs = []
    dcors = []
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
                    y_preds_logits = FF(features).squeeze()
                    y_preds = (torch.sign(y_preds_logits) + 1) / 2
                    train_acc = (torch.sum(y_preds == acc_data.y) / len(acc_data)).item()
                    dcorr = (get_dcorr(y_preds_logits[acc_data.y == 0], acc_data.cf[acc_data.y == 0])
                             + get_dcorr(y_preds_logits[acc_data.y == 1], acc_data.cf[acc_data.y == 1])) / 2
                    accs.append(train_acc)
                    dcors.append(dcorr)

                    plt.close()
                    plt.plot(range(iter_count+1), accs)
                    plt.plot(range(iter_count+1), dcors)

                    # plt.savefig("acc_per_iter.png")

                if (iter_count % show_every == 0):
                    if acc_data is not None:
                        print("iter =", iter_count, ", acc =", train_acc, ", dcorr =", dcorr)
                    print('Iter: {}, D: {:.4}, G:{:.4}'.format(iter_count, d_total_error.item(), g_error.item()))
                    print("loss: ", F.binary_cross_entropy_with_logits(preds, y).item())
                    print()
                    plt.show()
            iter_count += 1

    with open(f"{name}_dcors.pkl", "wb") as fp:
        pickle.dump(dcors, fp)
    with open(f"{name}_accs.pkl", "wb") as fp:
        pickle.dump(accs, fp)
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

                    eo_yzero = torch.abs(
                        F.relu(preds[(y == 0) & (cf >= 3)]).sum() - F.relu(preds[(y == 0) & (cf < 3)]).sum())
                    eo_yone = torch.abs(
                        F.relu(preds[(y == 1) & (cf < 4)]).sum() - F.relu(preds[(y == 1) & (cf >= 4)]).sum())

                    g_error = (eo_yzero + eo_yone) * l + F.binary_cross_entropy_with_logits(preds, y)
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
