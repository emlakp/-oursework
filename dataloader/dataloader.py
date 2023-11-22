from torch.utils.data import Dataset, DataLoader, random_split, Subset
import torchvision
from torchvision.transforms import *
import sklearn
from sklearn.preprocessing import LabelBinarizer
import numpy as np
import pandas as pd


class CustomImageDataset(Dataset):
    def __init__(
            self,
            annotations_file,
            img_dir,
            transform=None,
            target_transform=None,
            val=False):
        self.img_labels = pd.read_csv(annotations_file)[['image', 'species']]
        names = np.unique(pd.read_csv('/content/train.csv').species)
        names = [i for i in names if i !=
                 'kiler_whale' and i != 'bottlenose_dolpin']
        self.le = preprocessing.LabelBinarizer()
        self.le.fit(names)

        self.img_dir = img_dir
        if val:
            self.transform = transforms.Compose(
                [
                    transforms.ConvertImageDtype(
                        torch.float), torchvision.transforms.Normalize(
                        mean, std), torchvision.transforms.Resize(
                        (384, 380))])
        else:
            self.transform = Compose(
                [ConvertImageDtype(torch.float),
                 RandomApply(torch.nn.ModuleList([RandomAffine(degrees=(-10, 10))]), p=0.3),
                 RandomApply(torch.nn.ModuleList([GaussianBlur(kernel_size=(3, 7))]), p=0.4),
                 Normalize(mean, std),
                 Resize((384, 380)),
                 ])

        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if image.shape[0] == 1:
            image = image.expand(3, *image.shape[1:])
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        if label == 'kiler_whale':
            label = 'killer_whale'
        if label == 'bottlenose_dolpin':
            label = 'bottlenose_dolphin'

        label = le.transform([label])[0]
        return image, label


def split_dataloaders(train_set, val_set, train_size=0.8):
    num_train = len(train_set)
    indices = list(range(num_train))
    split = int(np.floor(train_size * num_train))
    np.random.shuffle(indices)
    train_idx, valid_idx = indices[:split], indices[split:]

    traindata = Subset(train_set, indices=train_idx)
    valdata = Subset(val_set, indices=valid_idx)

    return ({'train': DataLoader(traindata, batch_size=128, num_workers=4),
             'val': DataLoader(valdata, batch_size=128, num_workers=4)},
            {'train': 40826, 'val': 10207})
