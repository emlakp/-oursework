import math
import torch
import copy
import time
from torch import optim
from dataloader.augmentations import *


def adam_cosine_lin(
        model,
        lr=1e-3,
        weight_decay=0.0001,
        warm_up_iter=3,
        lr_min=1e-3,
        lr_max=1e-2,
        T_max=10):
    """

    :param model: model to optimize
    :param lr: learning rate
    :param weight_decay: weightdecay hyperparamter
    :param warm_up_iter: number of warm up iterations
    :param lr_min: minimal learning rate in cosine annealing
    :param lr_max: maximum learning rate in cosine annealing
    :param T_max: number of steps
    :return:
    """
    def lambda_0(curr_iter): return curr_iter / warm_up_iter if curr_iter < warm_up_iter else \
        (lr_min + 0.5 * (lr_max - lr_min) * (
            1.0 + math.cos((curr_iter - warm_up_iter) / (T_max - warm_up_iter) * math.pi))) / 0.1

    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_0)
    return optimizer, scheduler


def train_model(
        model,
        dataloaders,
        dataset_sizes,
        criterion,
        optimizer,
        scheduler,
        num_epochs=25,
        device='cpu'):
    """

    :param model: model to train
    :param dataloaders: dictionary with train and val dataloader
    :param dataset_sizes: dictionary with trainset and valset size
    :param criterion: loss function
    :param optimizer: optimizer
    :param scheduler: lr scheduler
    :param num_epochs: num of epochs for training
    :param device: accelerating device
    :return:
    """
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

                if 0.15 < mix_decision < 0.3:
                    inputs, targets_a, targets_b, lam = cutmix(
                        inputs, labels, 1.)
                    mixed = True
                if mix_decision <= 0.15:
                    inputs, targets_a, targets_b, lam = mixed_data(
                        inputs, labels, 0.2, cuda=True)
                    mixed = True

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)

                    if mixed:
                        loss = mixed_criterion(
                            criterion, outputs, targets_a, targets_b, lam)
                    else:
                        loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        scheduler.step()
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += sum(preds ==
                                        torch.argmax(labels.data, dim=1))

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
                save_model(
                    epoch + 1000,
                    model,
                    optimizer,
                    scheduler,
                    epoch_loss)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time()
    print(
        f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model
