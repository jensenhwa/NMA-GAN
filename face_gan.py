from multiprocessing import get_context

import numpy as np
import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torch import nn, distributed
from torch.utils.data import DataLoader, DistributedSampler

from datasets.face_data import FaceData
from datasets.yale_data import YaleData
from diffae.choices import TrainMode
from nma_gan import discriminator2v, discriminator3v, discriminator2, discriminator3, discriminator_loss, get_optimizer

np.random.seed(0)


def ema(source, target, decay):
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key].data.copy_(target_dict[key].data * decay +
                                    source_dict[key].data * (1 - decay))


class FaceGAN(LightningModule):
    def __init__(self, encoder, data_type, mode: str, v_len: int, l: float, g: float, r: float,
                 batch_size: int, lr: float, cv_fold=None, partial: int = 0):
        super().__init__()
        self.data_type = data_type
        self.mode = mode
        self.v_len = v_len
        self.l = l
        self.g = g
        self.r = r
        self.lr = lr
        self.partial = partial
        assert batch_size % get_world_size() == 0
        self.batch_size = batch_size // get_world_size()
        self.cv_fold = cv_fold
        if self.mode == 'y':
            self.D2 = discriminator2()
            self.D3 = discriminator3()
        else:
            self.mode = 'v' if partial == 0 else 'v_partial'
            self.D2 = discriminator2v(v_len=v_len-partial)
            self.D3 = discriminator3v(v_len=v_len-partial)
        self.encoder = encoder
        self.FF = nn.Sequential(
            nn.Linear(512, 1),
        )
        self.race_FF: torch.nn.Module = None

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
        if self.data_type == "face":
            self.train_data = FaceData(set="train", device=self.device, dimension=128, cv_fold=self.cv_fold)
            self.val_data = FaceData(set="val", device=self.device, dimension=128, cv_fold=self.cv_fold)
            self.FF = nn.Linear(self.v_len-self.partial, 1)
        elif self.data_type == "yale":
            self.train_data = YaleData(set="train", device=self.device, dimension=128, cv_fold=self.cv_fold)
            self.val_data = YaleData(set="val", device=self.device, dimension=128, cv_fold=self.cv_fold)
            self.FF = nn.Linear(self.v_len-self.partial, len(torch.unique(self.train_data.y)))
        else:
            raise ValueError("data type not supported")
        self.race_FF = nn.Linear(self.partial, len(torch.unique(self.train_data.cf)))
        print('train data:', len(self.train_data))
        print('val data:', len(self.val_data))

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
        x, y, cf = batch

        # train discriminator
        if optimizer_idx == 0:
            with torch.no_grad():
                features = self.encoder.model.encoder.forward(x)
                preds = self.FF(features[:, self.partial:]).squeeze()

            if self.mode == 'y':
                z_tilde = y

                y_prime = torch.rand((len(x),), device=x.device) * 0.5
                y_prime[y == 1] = y_prime[y == 1] + 0.5

                s = cf  # sensitive atts

                logits_real2 = self.D2(torch.stack((s, y_prime, z_tilde), dim=1))
                logits_fake2 = self.D2(torch.stack((s, F.sigmoid(preds), z_tilde), dim=1))

                logits_real3 = self.D3(torch.stack((y_prime, z_tilde), dim=1))
                logits_fake3 = self.D3(torch.stack((F.sigmoid(preds), z_tilde), dim=1))

            elif self.mode == 'v':
                z_tilde = y.unsqueeze(1)

                v_prime = features.detach().clone()
                for val in torch.unique(y):
                    v_prime[y == val] = v_prime[y == val][
                        torch.randint(v_prime[y == val].size()[0], (v_prime[y == val].size()[0],))]

                s = cf.unsqueeze(1)  # sensitive atts

                logits_real2 = self.D2(torch.cat((s, v_prime, z_tilde), dim=1))
                logits_fake2 = self.D2(torch.cat((s, features, z_tilde), dim=1))

                logits_real3 = self.D3(torch.cat((v_prime, z_tilde), dim=1))
                logits_fake3 = self.D3(torch.cat((features, z_tilde), dim=1))

            elif self.mode == 'v_partial':

                z_tilde = y.unsqueeze(1)

                v_prime = features[:, self.partial:].detach().clone()
                for val in torch.unique(y):
                    v_prime[y == val] = v_prime[y == val][
                        torch.randint(v_prime[y == val].size()[0], (v_prime[y == val].size()[0],))]

                s = cf.unsqueeze(1)  # sensitive atts

                logits_real2 = self.D2(torch.cat((s, v_prime, z_tilde), dim=1))
                logits_fake2 = self.D2(torch.cat((s, features[:, self.partial:], z_tilde), dim=1))

                logits_real3 = self.D3(torch.cat((v_prime, z_tilde), dim=1))
                logits_fake3 = self.D3(torch.cat((features[:, self.partial:], z_tilde), dim=1))
            else:
                raise ValueError("Unsupported mode")
            l2 = discriminator_loss(logits_real2, logits_fake2)
            l3 = discriminator_loss(logits_real3, logits_fake3)

            loss = l2 + l3
            self.log("d_loss", loss, prog_bar=True)
            return loss

        # train generator
        if optimizer_idx == 1:
            features = self.encoder.model.encoder.forward(x)
            preds = self.FF(features[:, self.partial:])

            if self.mode == 'y':
                z_tilde = y

                y_prime = torch.rand((len(x),), device=x.device) * 0.5
                y_prime[y == 1] = y_prime[y == 1] + 0.5

                s = cf

                logits_real2 = self.D2(torch.stack((s, y_prime, z_tilde), dim=1))
                logits_fake2 = self.D2(torch.stack((s, F.sigmoid(preds.squeeze()), z_tilde), dim=1))
                logits_real3 = self.D3(torch.stack((y_prime, z_tilde), dim=1))
                logits_fake3 = self.D3(torch.stack((F.sigmoid(preds.squeeze()), z_tilde), dim=1))
            elif self.mode == 'v':
                z_tilde = y.unsqueeze(1)

                v_prime = features.detach().clone()
                for val in torch.unique(y):
                    v_prime[y == val] = v_prime[y == val][
                        torch.randint(v_prime[y == val].size()[0], (v_prime[y == val].size()[0],))]

                s = cf.unsqueeze(1)

                logits_real2 = self.D2(torch.cat((s, v_prime, z_tilde), dim=1))
                logits_fake2 = self.D2(torch.cat((s, features, z_tilde), dim=1))
                logits_real3 = self.D3(torch.cat((v_prime, z_tilde), dim=1))
                logits_fake3 = self.D3(torch.cat((features, z_tilde), dim=1))
            elif self.mode == 'v_partial':
                z_tilde = y.unsqueeze(1)

                v_prime = features[:, self.partial:].detach().clone()
                for val in torch.unique(y):
                    v_prime[y == val] = v_prime[y == val][
                        torch.randint(v_prime[y == val].size()[0], (v_prime[y == val].size()[0],))]

                s = cf.unsqueeze(1)

                logits_real2 = self.D2(torch.cat((s, v_prime, z_tilde), dim=1))
                logits_fake2 = self.D2(torch.cat((s, features[:, self.partial:], z_tilde), dim=1))
                logits_real3 = self.D3(torch.cat((v_prime, z_tilde), dim=1))
                logits_fake3 = self.D3(torch.cat((features[:, self.partial:], z_tilde), dim=1))

                preds_race = self.race_FF(features[:, :self.partial])
                race_loss = F.cross_entropy(preds_race, cf.long())
                self.log("race_loss", race_loss)
            else:
                raise ValueError("Unsupported mode")

            l2 = discriminator_loss(logits_real2, logits_fake2)
            l3 = discriminator_loss(logits_real3, logits_fake3)

            ci_loss = (l2 - l3) ** 2

            # sample t's from a uniform distribution
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
                self.log('loss', losses['loss'])
                for key in ['vae', 'latent', 'mmd', 'chamfer', 'arg_cnt']:
                    if key in losses:
                        self.logger.experiment.add_scalar(
                            f'loss/{key}', losses[key], self.encoder.num_samples)
            if self.data_type == 'face':
                gender_loss = F.binary_cross_entropy_with_logits(preds.squeeze(), y)
            else:
                gender_loss = F.cross_entropy(preds, y.long())
            if self.mode == 'v_partial':
                loss = diffae_loss + self.l * ci_loss + self.g * gender_loss + self.r * race_loss
            else:
                loss = diffae_loss + self.l * ci_loss + self.g * gender_loss

            self.log("new_g_loss", loss, prog_bar=True)
            self.log("ci_loss", ci_loss)
            self.log("gender_loss", gender_loss)

            # To maintain compatibility with previous models
            self.log("g_loss", diffae_loss + self.l * ci_loss)
            self.log("ff_loss", gender_loss)
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
        if optimizer_idx == 1:
            self.encoder.on_before_optimizer_step(optimizer, optimizer_idx)

    def validation_step(self, batch, batch_idx):
        x, y, cf = batch
        features = self.encoder.encode(x)
        y_preds = self.FF(features[:, self.partial:])
        if self.data_type == 'face':
            y_preds = (torch.sign(y_preds.squeeze()) + 1) / 2
            one_hot = torch.nn.functional.one_hot(cf.long(), num_classes=len(torch.unique(self.train_data.cf))).bool()
            # accuracies by skin class
            train_accs = torch.sum(one_hot & (
                        y_preds.unsqueeze(1).expand(one_hot.shape) * one_hot == y.unsqueeze(1).expand(
                    one_hot.shape) * one_hot), dim=0) / torch.sum(one_hot, dim=0)
            self.log("per_skin_acc", {str(i): x for i, x in enumerate(train_accs)}, sync_dist=True)
        else:
            y_preds = torch.argmax(y_preds, dim=-1)
        train_acc = (torch.sum(y_preds == y) / len(x)).item()
        self.log("acc", train_acc, sync_dist=True)

    def configure_optimizers(self):
        D_solver = get_optimizer(nn.ModuleList([self.D2, self.D3]), lr=self.lr)
        G_solver = get_optimizer(nn.ModuleList([self.encoder, self.FF, self.race_FF]), lr=self.lr)
        return D_solver, G_solver


def get_world_size():
    if distributed.is_initialized():
        return distributed.get_world_size()
    else:
        return 1
