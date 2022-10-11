import os
import sys
from pathlib import Path

import numpy as np
import torch
from datasets.face_data import FaceDataForLatent128
from nma_gan import FaceGAN
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy

diffae_path = Path(".") / "diffae"
sys.path.append(str(diffae_path.resolve()))
from templates import LitModel
from templates_latent import ffhq128_autoenc_130M, ffhq128_autoenc_latent

def get_state_dict():
    conf = ffhq128_autoenc_130M()
    conf.T_eval = 100
    conf.latent_T_eval = 100
    # from choices import TrainMode
    # conf.train_mode = TrainMode.diffusion
    conf.base_dir = "checkpoints_jh2"
    # conf.pretrain.path = 'diffae/checkpoints/ffhq256_autoenc/last.ckpt'
    # conf.latent_infer_path = 'diffae/checkpoints/ffhq256_autoenc/latent.pkl'
    model = LitModel(conf)

    from nma_gan import FaceGAN
    gan = FaceGAN(model, 512, 10000, 64)

    state = torch.load(f'checkpoints_jh2/{conf.name}/last.ckpt', map_location='cpu')
    print(gan.load_state_dict(state['state_dict'], strict=False))
    return gan.encoder.state_dict()


if __name__ == "__main__":
    np.random.seed(8)
    conf = ffhq128_autoenc_latent()
    conf.T_eval = 100
    conf.latent_T_eval = 100
    # from choices import TrainMode
    # conf.train_mode = TrainMode.diffusion
    conf.base_dir = "checkpoints_jh3"
    conf.pretrain = None
    # junk (just to properly set conds_mean and std as buffers)
    conf.latent_infer_path = 'diffae/checkpoints/ffhq256_autoenc/latent.pkl'

    def make_dataset(path=None, **kwargs):
        return FaceDataForLatent128()

    conf.make_dataset = make_dataset
    model = LitModel(conf)
    model.conds = None

    print(model.load_state_dict(get_state_dict(), strict=False))


    gpus = [0]
    nodes = 1

    if not os.path.exists(conf.logdir):
        os.makedirs(conf.logdir)
    checkpoint = ModelCheckpoint(dirpath=f'{conf.logdir}',
                                 save_last=True,
                                 save_top_k=1,
                                 every_n_train_steps=conf.save_every_samples //
                                 conf.batch_size_effective)
    checkpoint_path = f'{conf.logdir}/last.ckpt'
    print('ckpt path:', checkpoint_path)
    if os.path.exists(checkpoint_path):
        resume = checkpoint_path
        print('resume!')
    else:
        if conf.continue_from is not None:
            # continue from a checkpoint
            resume = conf.continue_from.path
        else:
            resume = None

    tb_logger = pl_loggers.TensorBoardLogger(save_dir=conf.logdir,
                                             name=None,
                                             version='')

    if len(gpus) == 1 and nodes == 1:
        strategy = None
    else:
        strategy = DDPStrategy(find_unused_parameters=True)

    print("training now")
    trainer = Trainer(
        max_steps=conf.total_samples // conf.batch_size_effective,
    #     resume_from_checkpoint=resume,
        gpus=gpus,
    #     num_nodes=nodes,
        accelerator="gpu",
        strategy=strategy,
        precision=16 if conf.fp16 else 32,
        callbacks=[
            checkpoint,
            LearningRateMonitor(),
        ],
        # clip in the model instead
        # gradient_clip_val=conf.grad_clip,
        replace_sampler_ddp=True,
        logger=tb_logger,
        accumulate_grad_batches=conf.accum_batches,
    )

    trainer.fit(model)
