import torch
import numpy as np
import os


def setup_seed(seed: int):
    """Fixes random seed for reproducibility

    :param seed:
    :return:
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(12)
