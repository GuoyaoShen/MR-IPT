import torch

import utils.ipt_util as utility
from utils.evaluation_utils import NMSE, PSNR, SSIM
from tqdm.autonotebook import tqdm
from help_func import print_var_detail
import torch.nn.functional as F
from time import time


class TrainerMulti:
    """
    trainer for MR-IPT model, includes train() and test() function

    Args:
    ----------
    loader_train: dataloader
        dataloader for training, expected tuple consists of pairs
    loader_test: dataloader
        dataloader for testing, expected tuple consists of pairs
    my_model: model
        ipt model with multi-heads/tails
    my_loss: nn.modules.loss._Loss
        loss modules defined in loss._init_
    PATH_MODEL : string
        path to saved model
    if_save:
        whether to save model
    """

    def __init__(self, loader_train, loader_test, my_model, my_loss, optimizer,
                 PATH_MODEL, device, NUM_EPOCH=50, RESUME_EPOCH=0, total_batch_train=None, if_save=True):
        self.loader_train = loader_train  # list [image, filename, slice_index]
        self.loader_test = loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = optimizer
        self.PATH_MODEL = PATH_MODEL
        self.cpu = False
        if RESUME_EPOCH > 0:
            print('Load checkpoint from:', PATH_MODEL + 'model_E' + str(RESUME_EPOCH) + '.pt')
            self.model.load_state_dict(torch.load(PATH_MODEL + 'model_E' + str(RESUME_EPOCH) + '.pt')['model_state_dict'])
            self.optimizer.load_state_dict(
                torch.load(PATH_MODEL + 'model_E' + str(RESUME_EPOCH) + '.pt')['optimizer_state_dict'])

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
        if total_batch_train == None:
            self.total_batch_train = len(loader_train)
        else:
            self.total_batch_train = total_batch_train

    def train(self, show_step=-1, show_test=True):
        self.model = self.model.to(self.device)
        optimizer_to(self.optimizer, self.device)
        self.model.train()

        timer_data, timer_model = utility.timer(), utility.timer()

        # training iteration
        pbar = tqdm(range(self.RESUME_EPOCH, self.NUM_EPOCH), desc='LOSS')
        for i in pbar:
            self.running_loss_train = 0
            num_nan = 0
            start = time()
            for batch, (images, levels, filename, idx, mask, types) in enumerate(self.loader_train):
                if batch == self.total_batch_train:
                    break

                if batch % 1000 == 0 or batch < 5:
                    end = time()
                    print("trained batch num: ", batch, " of total ", len(self.loader_train))
                    print("elapsed time: ", end - start)
                    start = time()

                images_input = images[0].to(self.device)  # [pair]
                images_target = images[1].to(self.device)  # [pair]
                levels = levels.to(self.device)
                types = types.to(self.device)
                if i == 0 and batch == 0:
                    print_var_detail(images_input, "images_input")
                    print_var_detail(images_target, "images_target")
                    print_var_detail(levels, 'levels')
                    print_var_detail(types, 'types')
                    timer_data.hold()
                    timer_model.tic()
                timer_data.hold()
                timer_model.tic()

                self.optimizer.zero_grad()
                sr = self.model(x=images_input, levels=levels, types=types)

                loss = self.loss(sr, images_target)
                if torch.isnan(loss):
                    print("nan loss occur")
                    print(filename)
                    num_nan += 1
                else:
                    loss.backward()
                    self.optimizer.step()
                    self.running_loss_train += loss.item()
                timer_model.hold()
                timer_data.tic()
            self.running_loss_train /= (len(self.loader_train) - num_nan)

            pbar.set_description("Loss=%f" % (self.running_loss_train))
            if show_step > 0:
                if (i + 1) % show_step == 0:
                    print('*** EPOCH ' + str(i + 1) + ' || AVG LOSS: ' + str(self.running_loss_train))

            # save model
            if self.if_save:
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                }, self.PATH_MODEL + 'model_E' + str(i + 1) + '.pt')
                print('MODEL SAVED at epoch: ' + str(i + 1))

        # save model
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, self.PATH_MODEL + 'model_latest.pt')
        print('MODEL SAVED')
        # test model
        if show_test:
            loss_test, nmse, psnr, ssim = self.test()


        return self.model

    def prepare(self, *args):
        device = torch.device('cpu' if self.cpu else 'cuda')

        def _prepare(tensor):
            # if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(a) for a in args]

    def test(self):
        '''
        Test the reconstruction performance.
        '''
        self.model = self.model.to(self.device)
        self.model.eval()

        self.running_loss_test = 0.0
        nmse = 0.0
        psnr = 0.0
        ssim = 0.0
        num_nan = 0
        with torch.no_grad():
            for batch, (images, levels, filename, idx, mask, types) in enumerate(self.loader_test):
                if batch % 1000 == 0:
                    print("tested batch num: ", batch, " of total ", len(self.loader_test))
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
                psnr += psnr_(pred, tg, data_range=torch.ones(tg.shape[0]).to(self.device))
                ssim += ssim_(pred, tg, data_range=torch.ones(tg.shape[0]).to(self.device))

            nmse /= len(self.loader_test)
            psnr /= len(self.loader_test)
            ssim /= len(self.loader_test)

            self.running_loss_test /= (len(self.loader_test) - num_nan)

        print('### TEST LOSS: ',
              str(self.running_loss_test) + '|| NMSE: ' + str(nmse) + '|| PSNR: ' + str(psnr) + '|| SSIM: ' + str(ssim))
        print('----------------------------------------------------------------------')

        return self.running_loss_test, nmse, psnr, ssim

def optimizer_to(optim, device):
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
