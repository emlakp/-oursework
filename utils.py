import torch
from torch.utils.data import Dataset
from torchvision import models
from torchvision.transforms import *
from sklearn.preprocessing import LabelBinarizer
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split, Subset
import numpy as np, pandas as pd, os

def setup_seed(seed):
  os.environ['PYTHONHASHSEED'] = str(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  np.random.seed(seed)
  torch.backends.cudnn.deterministic = True

setup_seed(12)

def weights_init_normal(model):
  classnames = model.__class__.__name__
  if classnames.find('Linear'):
    y = model.in_features
    model.weight.data.normal(0.0,1/np.sqrt(y))
    model.bias.data.fill_(0)


class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)[['image', 'species']]
        names = np.unique(pd.read_csv('/content/train.csv').species)
        names = [i for i in names if i != 'kiler_whale' and i != 'bottlenose_dolpin']
        self.le = preprocessing.LabelBinarizer()
        self.le.fit(names)

        self.img_dir = img_dir
        self.transform = transform
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


def mixed_data(x, y, alpha=1.0, cuda=True):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if cuda:
        index = torch.randperm(batch_size).to(device)
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam


def mixed_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def rand_bbox(size, lam):
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

def cutmix(data, target, alpha):
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_target = target[indices]

    lam = np.clip(np.random.beta(alpha, alpha),0.3,0.4)
    bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)
    new_data = data.clone()
    new_data[:, :, bby1:bby2, bbx1:bbx2] = data[indices, :, bby1:bby2, bbx1:bbx2]
    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-1] * data.size()[-2]))

    return new_data, target,shuffled_target,lam


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:

                mixed = False

                inputs = inputs.to(device)
                labels = labels.to(device).float()

                mix_decision = np.random.rand()

                # if mix_decision > 0.15 and mix_decision < 0.3:
                #  inputs, targets_a, targets_b, lam = cutmix(inputs, labels, 1.)
                #  mixed = True
                # if mix_decision <= 0.15:
                #  inputs, targets_a, targets_b, lam = mixed_data(inputs, labels,
                #                                       0.2, cuda = True)
                #  mixed = True

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)

                    # if mixed:
                    #    loss = mixed_criterion(criterion, outputs, targets_a, targets_b, lam)
                    # else:
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        scheduler.step()
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += sum(preds == torch.argmax(labels.data, dim=1))

                print(loss.item() * inputs.size(0))
                print(sum(preds == torch.argmax(labels.data, dim=1)) / len(preds))

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            if phase == 'train':
                train_losses.append(epoch_loss)
                train_accs.append(epoch_acc)
            else:
                val_losses.append(epoch_loss)
                val_accs.append(epoch_acc)

            if phase == 'val':
                save_model(epoch + 1000, model, optimizer, scheduler, epoch_loss)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time()
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model