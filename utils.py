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

def get_state_dict(self, *args, **kwargs):
    """Obtains state dict for fetching model weights. Quick workaround for hashing error

    :param self:
    :param args:
    :param kwargs:
    :return:
    """
    kwargs.pop("check_hash")
    return load_state_dict_from_url(self.url, *args, **kwargs)

WeightsEnum.get_state_dict = get_state_dict


setup_seed(12)
