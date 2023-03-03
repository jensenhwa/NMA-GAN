import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy

from face_gan import FaceGAN

diffae_path = Path(".") / "diffae"
sys.path.append(str(diffae_path.resolve()))
from templates import LitModel
from templates import ffhq128_autoenc_130M


def main(args):

    if args.cv:
        # do 5 fold cross validation
        base_dir = [args.dir + "/" + str(i) for i in range(5)]
        cv_fold = range(5)
    else:
        base_dir = [args.dir]
        cv_fold = [None]

    for dir, cv in zip(base_dir, cv_fold):
        np.random.seed(8)
        conf = ffhq128_autoenc_130M()
        conf.T_eval = 100
        conf.latent_T_eval = 100
        # from choices import TrainMode
        # conf.train_mode = TrainMode.diffusion
        conf.base_dir = dir  # "checkpoints_jh2"
        # conf.pretrain.path = 'diffae/checkpoints/ffhq128_autoenc/last.ckpt'
        # conf.latent_infer_path = 'diffae/checkpoints/ffhq128_autoenc/latent.pkl'
        model = LitModel(conf)
        state = torch.load(f'diffae/checkpoints/{conf.name}/last.ckpt', map_location='cpu')
        print(model.load_state_dict(state['state_dict'], strict=False))

        gpus = [0, 1, 2, 3]
        nodes = 1
        gan = FaceGAN(
            model,
            "yale",
            512,
            args.l,
            args.g,
            args.r,
            args.batch_size,
            args.lr,
            cv_fold=cv,
            partial=args.partial
        )

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

        wandb_logger = WandbLogger(
            project="nma-gan",
            entity="jphwa",
            config=args,
            save_dir=conf.logdir)

        if len(gpus) == 1 and nodes == 1:
            strategy = None
        else:
            strategy = DDPStrategy(find_unused_parameters=True)

        print("training now")
        trainer = Trainer(
            max_epochs=1000,
            resume_from_checkpoint=resume,
            gpus=gpus,
            num_nodes=nodes,
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
            logger=wandb_logger,
            accumulate_grad_batches=conf.accum_batches,
        )

        trainer.fit(gan)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "--l",
        required=True,
        type=float,
        help="CI regularization strength"
    )
    parser.add_argument(
        "--g",
        required=True,
        type=float,
        help="sex prediction strength"
    )
    parser.add_argument(
        "--r",
        required=True,
        type=float,
        help="skin type prediction strength"
    )
    parser.add_argument(
        "--lr",
        required=True,
        type=float,
        help="learning rate"
    )
    parser.add_argument(
        "--batch_size",
        required=True,
        type=int,
        default=32,
        help="batch size"
    )

    parser.add_argument(
        "--dir",
        default="run_" + datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
        help="model save directory"
    )
    parser.add_argument(
        "--cv",
        action="store_true",
        help="perform 5-fold cross validation"
    )
    parser.add_argument(
        "--partial",
        type=int,
        help="use partial v-space architecture"
    )

    args = parser.parse_args()

    main(args)
