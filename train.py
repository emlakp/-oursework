from dataloader.augmentations import *
from dataloader.dataloader import *
from model.model import get_model
from training.trainer import *

if __name__ == '__main__':
    model = get_model(
        'efficientnet_b3',
        pretrained=True,
        frozen=True,
        device='cpu')

    warm_up_iter = 3
    lr_min = 1e-3
    lr_max = 1e-2
    T_max = 10

    optimizer, scheduler = adam_cosine_lin(
        model, warm_up_iter, lr_min, lr_max, T_max)
    criterion = torch.nn.CrossEntropyLoss()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_losses = []
    val_accs = []
    train_losses = []
    train_accs = []

    mean = torch.Tensor([51.7598, 57.7707, 64.2816])  # precomputed values
    std = torch.Tensor([30.0926, 30.4241, 31.7076])  # precomputed values

    train_set = CustomImageDataset('./train.csv', './train_images', val=False)
    val_set = CustomImageDataset('./train.csv', './train_images', val=True)

    dataloaders, dataset_sizes = split_dataloaders(train_set, val_set, 0.8)

    model = train_model(model, dataloaders, dataset_sizes,
                        criterion, optimizer, scheduler, 20, device)
