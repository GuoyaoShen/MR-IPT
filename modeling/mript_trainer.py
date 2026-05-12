"""Training and evaluation utilities for MR-IPT."""

import torch

import utils.ipt_util as utility
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
        RESUME_EPOCH: Resume epoch index. If > 0, loads checkpoint
            ``model_E{RESUME_EPOCH}.pt``.
        total_batch_train: Optional cap on train batches per epoch.
        if_save: Whether to save checkpoints during/after training.
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
        NUM_EPOCH=50,
        RESUME_EPOCH=0,
        total_batch_train=None,
        if_save=True,
    ):
        self.loader_train = loader_train
        self.loader_test = loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = optimizer
        self.PATH_MODEL = PATH_MODEL
        self.cpu = False
        if RESUME_EPOCH > 0:
            print(
                f"Load checkpoint from: {PATH_MODEL + 'model_E' + str(RESUME_EPOCH) + '.pt'}"
            )
            self.model.load_state_dict(
                torch.load(PATH_MODEL + "model_E" + str(RESUME_EPOCH) + ".pt")[
                    "model_state_dict"
                ]
            )
            self.optimizer.load_state_dict(
                torch.load(PATH_MODEL + "model_E" + str(RESUME_EPOCH) + ".pt")[
                    "optimizer_state_dict"
                ]
            )

        self.RESUME_EPOCH = RESUME_EPOCH
        self.NUM_EPOCH = NUM_EPOCH
        self.device = device
        self.error_last = 1e8
        self.nmse = 0
        self.psnr = 0
        self.ssim = 0
        self.running_loss_train = 0
        self.running_loss_test = 0
        self.nan_sr = None
        self.nan_images_target = None
        self.if_save = if_save
        self.best_val_loss = float("inf")
        self.best_epoch = -1
        if total_batch_train is None:
            self.total_batch_train = len(loader_train)
        else:
            self.total_batch_train = total_batch_train

    def train(self, show_step=-1, show_test=True):
        """Run training and optional validation.

        Args:
            show_step: If > 0, prints average train loss every ``show_step``
                epochs.
            show_test: Whether to run ``test()`` at the end of each epoch.

        Returns:
            The trained model instance.
        """
        self.model = self.model.to(self.device)
        optimizer_to(self.optimizer, self.device)
        self.model.train()

        timer_data, timer_model = utility.timer(), utility.timer()

        # training iteration
        pbar = tqdm(range(self.RESUME_EPOCH, self.NUM_EPOCH), desc="LOSS")
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
                if batch == self.total_batch_train:
                    break

                images_input = images[0].to(self.device)
                images_target = images[1].to(self.device)
                levels = levels.to(self.device)
                types = types.to(self.device)
                if i == 0 and batch == 0:
                    print_var_detail(images_input, "images_input")
                    print_var_detail(images_target, "images_target")
                    print_var_detail(levels, "levels")
                    print_var_detail(types, "types")
                    timer_data.hold()
                    timer_model.tic()
                timer_data.hold()
                timer_model.tic()

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
                        f"Epoch {i + 1}/{self.NUM_EPOCH} Batch {batch + 1}/{self.total_batch_train} Loss {loss.item():.6f}"
                    )
                timer_model.hold()
                timer_data.tic()
            self.running_loss_train /= len(self.loader_train) - num_nan

            pbar.set_description(f"Loss={self.running_loss_train:f}")
            if show_step > 0 and (i + 1) % show_step == 0:
                print(
                    f"*** EPOCH {i + 1} || AVG LOSS: {self.running_loss_train:.6f}"
                )

            # save model
            if self.if_save:
                torch.save(
                    {
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                    },
                    self.PATH_MODEL + "model_E" + str(i + 1) + ".pt",
                )
                print(f"MODEL SAVED at epoch: {i + 1}")

            # test model every epoch and track the best validation loss
            if show_test:
                loss_test, nmse, psnr, ssim = self.test()
                if loss_test < self.best_val_loss:
                    self.best_val_loss = loss_test
                    self.best_epoch = i + 1
                    if self.if_save:
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
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            self.PATH_MODEL + "model_latest.pt",
        )
        print("MODEL SAVED")
        if show_test and self.best_epoch > 0:
            print(
                f"BEST EPOCH: {self.best_epoch} || BEST VAL LOSS: {self.best_val_loss:.6f}"
            )

        return self.model

    def prepare(self, *args):
        """Move tensors to the trainer-selected device.

        Args:
            *args: Input tensors.

        Returns:
            List of tensors moved to CPU or CUDA based on ``self.cpu``.
        """
        device = torch.device("cpu" if self.cpu else "cuda")

        def _prepare(tensor):
            # if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(a) for a in args]

    def test(self):
        """Evaluate reconstruction performance on ``loader_test``.

        Returns:
            Tuple of ``(running_loss_test, nmse, psnr, ssim)``.
        """
        self.model = self.model.to(self.device)
        self.model.eval()

        self.running_loss_test = 0.0
        nmse = 0.0
        psnr = 0.0
        ssim = 0.0
        num_nan = 0
        with torch.no_grad():
            for batch, (images, levels, _filename, _idx, _mask, types) in enumerate(
                self.loader_test
            ):
                if batch % 1000 == 0:
                    print(f"tested batch num: {batch} of total {len(self.loader_test)}")
                timer_data, timer_model = utility.timer(), utility.timer()
                images_input = images[0].to(self.device)  # [pair]
                images_target = images[1].to(self.device)  # [pair]
                levels = levels.to(self.device)
                types = types.to(self.device)
                timer_data.hold()
                timer_model.tic()

                sr = self.model(x=images_input, levels=levels, types=types)
                loss = self.loss(sr, images_target)
                if torch.isnan(loss):
                    num_nan += 1
                else:
                    self.running_loss_test += loss.item()
                timer_model.hold()
                timer_data.tic()

                # evaluation metrics
                tg = images_target.detach()
                pred = sr.detach()  # [B,C(1),H,W]

                # print('pred.shape:', pred.shape)
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
        print("----------------------------------------------------------------------")

        return self.running_loss_test, nmse, psnr, ssim


def optimizer_to(optim, device):
    """Move optimizer state tensors to a target device.

    Args:
        optim: Torch optimizer instance.
        device: Target torch device.
    """
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)
