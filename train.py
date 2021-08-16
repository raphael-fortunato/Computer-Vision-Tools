import cv2
import seaborn as sns
import time
import os
import sys
import copy
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split


import matplotlib
import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')
from utils.models import get_resnet50, get_vgg19
from utils.args import classification_args

# training the model
def train_model(
        args,
        model,
        dataloaders,
        optimizer,
        scheduler,
        criterion):

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(args.epochs):
        print('Epoch {}/{}'.format(epoch, args.epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            y_true, y_pred = [], []

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                # move labels and inputs to correct device
                inputs = inputs.to(device)
                labels = labels.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs, features = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                # if loss is infinite end training
                if not math.isfinite(loss):
                    print("Loss is {}, stopping training".format(loss))
                    sys.exit(1)

                # statistics
                running_loss += loss.item() * inputs.size(0)
                y_true.extend(labels.detach().cpu().tolist())
                y_pred.extend(outputs.argmax(1).detach().cpu().tolist())

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            # The average recall over all the classes
            epoch_acc = balanced_accuracy_score(y_true, y_pred)
            if phase == 'val':
                # change learning rate based on val loss
                if args.lr_scheduler == 'plateau':
                    scheduler.step(epoch_loss)
                else:
                    scheduler.step()
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    torch.save(best_model_wts, f'models/{args.model}_{time.time()}.pt')
    model.load_state_dict(best_model_wts)
    return model


def get_transforms(args, train):
    if train:
        transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize((args.img_size, args.img_size)),
                    transforms.ColorJitter(.1, .1, .1),
                    transforms.RandomCrop(args.img_size * .8),
                    transforms.RandomHorizontalFlip(p=.5),
                    transforms.RandomRotation(90),
                    transforms.Normalize(
                        [0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225])
                    ])
    else:
        transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize((args.img_size, args.img_size)),
                    transforms.Normalize(
                        [0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225])
                    ])
    return transform


def get_dataloaders(args, train):
    train_dataset = torchvision.datasets.ImageFolder(args.root, transform=transforms)
    train_dataset = torchvision.datasets.ImageFolder(args.root, transform=transforms)

    len_dataset = len(dataset)
    indices = list(range(len_dataset))
    split = int(np.floor(.9 * num_train))

    if args.shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    # load dataset
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, sampler=valid_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    dataloaders = {"train": train_loader,
                    "val": valid_loader}
    return dataloaders


if __name__ == "__main__":
    # retreiving command line arguments
    args = classification_args()
    for arg in vars(args):
        print(arg, getattr(args, arg))
    # setting seeds for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # define classification model
    if args.model == 'resnet':
        model = get_resnet50(args.path)
    elif args.model == 'vgg':
        model = get_vgg19(args.path)
    else:
        print("invalid model name")
        sys.exit(1)
    model.to(device)

    # define optimzer and scheduler
    criterion = nn.CrossEntropyLoss(),

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(
            params,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay)

    if args.lr_scheduler == 'plateau':
        # Decay LR by a factor of 0.1 every 15 epochs
        scheduler = lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=args.gamma,
                patience=args.patience, verbose=True)
    elif args.lr_scheduler == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(
                optimizer,
                args.step_size)

    dataloaders = get_dataloaders(args)

    model = train_model(
            args,
            model,
            dataloaders,
            optimizer,
            scheduler,
            criterion)
