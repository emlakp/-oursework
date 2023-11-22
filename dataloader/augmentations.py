import numpy as np
import torch


def mixed_data(
        x: torch.Tensor,
        y: torch.Tensor,
        alpha: float = 1.0,
        cuda: bool = True):
    """Implements MixUp augmentation


    :param x: features
    :param y: target
    :param alpha: mixing hyperparameter
    :param cuda: accelerating device
    :return: mixed features, targets for initial mixed images, mixing coefficient
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if cuda:
        index = torch.randperm(batch_size).to(torch.device('cuda'))
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam


def mixed_criterion(
        criterion: torch.nn.modules.loss,
        pred: torch.Tensor,
        y_a: torch.Tensor,
        y_b: torch.Tensor,
        lam: float):
    """Calculates the mixed loss for augmented instances

    :param criterion:
    :param pred:
    :param y_a:
    :param y_b:
    :param lam:
    :return: mixed loss
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def rand_bbox(size, lam):
    """Function to produce 2 random crops

    :param size:
    :param lam:
    :return: coordinates of cropped rectangles
    """
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2


def cutmix(data: torch.Tensor, target: torch.Tensor, alpha: float):
    """Implements CutMix augmentation

    :param data: features
    :param target: labels
    :param alpha: mixing hyperparameter
    :return: augmented sampels
    """
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_target = target[indices]

    lam = np.clip(np.random.beta(alpha, alpha), 0.3, 0.4)
    bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)
    new_data = data.clone()
    new_data[:,
             :,
             bby1:bby2,
             bbx1:bbx2] = data[indices,
                               :,
                               bby1:bby2,
                               bbx1:bbx2]
    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) /
               (data.size()[-1] * data.size()[-2]))

    return new_data, target, shuffled_target, lam
