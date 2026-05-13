"""Training and evaluation utilities for MR-IPT."""

import torch

from utils.evaluation_utils import NMSE, PSNR, SSIM
from tqdm.autonotebook import tqdm
from utils.help_func import print_var_detail


class TrainerMulti:
    """Trainer for MR-IPT with train and validation routines.

    Args:
        loader_train: Training dataloader. Each batch is expected as
            ``(images, levels, filename, idx, mask, types)``.
        loader_test: Validation/test dataloader with the same batch format.
        my_model: MR-IPT model instance.
        my_loss: Loss function module.
        optimizer: Optimizer used for training.
        PATH_MODEL: Directory path prefix for checkpoint files.
        device: Torch device used for model/data execution.
        NUM_EPOCH: Total number of training epochs.
    """

    def __init__(
        self,
        loader_train,
        loader_test,
        my_model,
        my_loss,
        optimizer,
        PATH_MODEL,
        device,
        NUM_EPOCH=10,
    ):
        self.loader_train = loader_train
        self.loader_test = loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = optimizer
        self.PATH_MODEL = PATH_MODEL
        self.NUM_EPOCH = NUM_EPOCH
        self.device = device
        self.running_loss_train = 0
        self.running_loss_test = 0
        self.best_val_loss = float("inf")
        self.best_epoch = -1

    def train(self):
        """Run training and validation.

        Returns:
            The trained model instance.
        """
        self.model.train()

        # training iteration
        pbar = tqdm(range(self.NUM_EPOCH), desc="LOSS")
        latest_val_loss = float("inf")
        for i in pbar:
            self.running_loss_train = 0
            num_nan = 0
            pbar_batch = tqdm(
                enumerate(self.loader_train),
                total=len(self.loader_train),
                desc=f"Epoch {i + 1}/{self.NUM_EPOCH}",
                leave=False,
            )
            for batch, (images, levels, filename, _idx, _mask, types) in pbar_batch:
                images_input = images[0].to(self.device)
                images_target = images[1].to(self.device)
                levels = levels.to(self.device)
                types = types.to(self.device)
                if i == 0 and batch == 0:
                    print_var_detail(images_input, "images_input")
                    print_var_detail(images_target, "images_target")
                    print_var_detail(levels, "levels")
                    print_var_detail(types, "types")

                self.optimizer.zero_grad()
                sr = self.model(x=images_input, levels=levels, types=types)

                loss = self.loss(sr, images_target)
                if torch.isnan(loss):
                    print("nan loss occur")
                    print(f"{filename}")
                    num_nan += 1
                else:
                    loss.backward()
                    self.optimizer.step()
                    self.running_loss_train += loss.item()
                    pbar_batch.set_description(
                        f"Epoch {i + 1}/{self.NUM_EPOCH} Batch {batch + 1}/{len(self.loader_train)} Loss {loss.item():.6f}"
                    )
            self.running_loss_train /= len(self.loader_train) - num_nan

            pbar.set_description(f"Loss={self.running_loss_train:f}")
            print(f"*** EPOCH {i + 1} || AVG LOSS: {self.running_loss_train:.6f}")

            # test every epoch and track best validation loss
            loss_test, nmse, psnr, ssim = self.test()
            latest_val_loss = loss_test
            if loss_test < self.best_val_loss:
                self.best_val_loss = loss_test
                self.best_epoch = i + 1
                nmse_v = nmse.item() if torch.is_tensor(nmse) else float(nmse)
                psnr_v = psnr.item() if torch.is_tensor(psnr) else float(psnr)
                ssim_v = ssim.item() if torch.is_tensor(ssim) else float(ssim)
                torch.save(
                    {
                        "epoch": i + 1,
                        "best_val_loss": self.best_val_loss,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                    },
                    self.PATH_MODEL + "model_best.pt",
                )
                print(
                    f"BEST MODEL SAVED at epoch: {i + 1} || VAL LOSS: {self.best_val_loss:.6f} "
                    f"|| NMSE: {nmse_v:.6f} || PSNR: {psnr_v:.6f} || SSIM: {ssim_v:.6f}"
                )

        # save model
        torch.save(
            {
                "epoch": self.NUM_EPOCH,
                "val_loss": latest_val_loss,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            self.PATH_MODEL + "model_latest.pt",
        )
        print("MODEL SAVED")
        if self.best_epoch > 0:
            print(
                f"BEST EPOCH: {self.best_epoch} || BEST VAL LOSS: {self.best_val_loss:.6f}"
            )
        print("-" * 30)

        return self.model

    def test(self):
        """Evaluate reconstruction performance on ``loader_test``.

        Returns:
            Tuple of ``(running_loss_test, nmse, psnr, ssim)``.
        """
        self.model.eval()

        self.running_loss_test = 0.0
        nmse = 0.0
        psnr = 0.0
        ssim = 0.0
        num_nan = 0
        with torch.no_grad():
            for _batch, (images, levels, _filename, _idx, _mask, types) in enumerate(
                self.loader_test
            ):
                images_input = images[0].to(self.device)  # [pair]
                images_target = images[1].to(self.device)  # [pair]
                levels = levels.to(self.device)
                types = types.to(self.device)

                sr = self.model(x=images_input, levels=levels, types=types)
                loss = self.loss(sr, images_target)
                if torch.isnan(loss):
                    num_nan += 1
                else:
                    self.running_loss_test += loss.item()

                # evaluation metrics
                tg = images_target.detach()
                pred = sr.detach()  # [B,C(1),H,W]

                nmse_ = NMSE().to(self.device)
                ssim_ = SSIM().to(self.device)
                psnr_ = PSNR().to(self.device)
                nmse += nmse_(pred, tg)
                psnr += psnr_(
                    pred, tg, data_range=torch.ones(tg.shape[0]).to(self.device)
                )
                ssim += ssim_(
                    pred, tg, data_range=torch.ones(tg.shape[0]).to(self.device)
                )

            nmse /= len(self.loader_test)
            psnr /= len(self.loader_test)
            ssim /= len(self.loader_test)

            self.running_loss_test /= len(self.loader_test) - num_nan

        nmse_v = nmse.item() if torch.is_tensor(nmse) else float(nmse)
        psnr_v = psnr.item() if torch.is_tensor(psnr) else float(psnr)
        ssim_v = ssim.item() if torch.is_tensor(ssim) else float(ssim)
        print(
            f"### TEST LOSS: {self.running_loss_test:.6f} || NMSE: {nmse_v:.6f} || PSNR: {psnr_v:.6f} || SSIM: {ssim_v:.6f}"
        )

        return self.running_loss_test, nmse, psnr, ssim
