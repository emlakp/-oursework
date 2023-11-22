import torch
from torch.utils.data import Dataset
from torchvision import models
from torchvision.transforms import *
from sklearn.preprocessing import LabelBinarizer
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split, Subset
import numpy as np
import pandas as pd
import os
import copy


def setup_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(12)
