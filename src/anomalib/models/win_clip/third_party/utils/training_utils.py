import random
import shutil
import time
import torch
# from torch.utils.tensorboard import SummaryWriter

from utils.visualization import *
from loguru import logger

def get_optimizer_from_args(model, lr, weight_decay, **kwargs) -> torch.optim.Optimizer:
    return torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr,
                             weight_decay=weight_decay)


def get_lr_schedule(optimizer):
    return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_dir_from_args(root_dir, class_name, **kwargs):

    exp_name = f"{kwargs['dataset']}-k-{kwargs['k_shot']}"

    csv_dir = os.path.join(root_dir, 'csv')
    csv_path = os.path.join(csv_dir, f"{exp_name}-indx-{kwargs['experiment_indx']}.csv")

    model_dir = os.path.join(root_dir, exp_name, 'models')
    img_dir = os.path.join(root_dir, exp_name, 'imgs')

    logger_dir = os.path.join(root_dir, exp_name, 'logger', class_name)

    log_file_name = os.path.join(logger_dir,
                                 f'log_{time.strftime("%Y-%m-%d-%H-%I-%S", time.localtime(time.time()))}.log')

    model_name = f'{class_name}'

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(logger_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)

    logger.start(log_file_name)

    logger.info(f"===> Root dir for this experiment: {logger_dir}")

    return model_dir, img_dir, logger_dir, model_name, csv_path
