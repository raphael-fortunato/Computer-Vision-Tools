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
# matplotlib.use('tkagg')
import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')
from utils.dataloaders import get_dataloaders
from utils.models import get_resnet50, get_vgg19
from utils.pipeline import SegmentationModel
from utils.args import classification_args
from utils.mutual_channel_loss import MCLoss
from utils.pairwise_conf_loss import PCLoss


# test the models performance
def test_model(args, model, dataloader, criterion):
    model.eval()
    running_loss = 0
    y_true, y_pred = [], []
    count = 0

    # Iterate over data.
    for inputs, labels in dataloader:
        # transfer labels and input to the correct device
        inputs = inputs.to(device)
        labels = labels.to(device)

        # forward
        with torch.no_grad():
            outputs, features = model(inputs)
            loss = criterion(outputs, labels)

        # statistics
        running_loss += loss.item() * inputs.size(0)
        y_true.extend(labels.detach().cpu().tolist())
        y_pred.extend(outputs.argmax(1).detach().cpu().tolist())

    epoch_loss = running_loss / len(dataloader.dataset)
    # balanced_accuracy_score to correct for the nonuniform class distribution
    epoch_acc = balanced_accuracy_score(y_true, y_pred)
    print('Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

    conf_matrix = confusion_matrix(y_true, y_pred)
    with np.errstate(divide='ignore', invalid='ignore'):
        per_class = np.diag(conf_matrix) / conf_matrix.sum(axis=1)
    print([f"{c}: {p}" for c, p in zip(CLASS_NAMES, per_class)])
    heatmap = sns.heatmap(
        conf_matrix / np.repeat(np.sum(conf_matrix, axis=1),16).reshape((16,16)),
        annot=True,
        fmt=".0%",
        cbar=False,
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES)
    plt.figure(figsize=(20, 15))
    figure = heatmap.get_figure()
    figure.savefig(
            f'heatmap_{args.model}_{time.time()}.png',
            dpi=700,
            bbox_inches='tight')


def get_transforms(args):
        transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize((args.img_size, args.img_size)),
                    transforms.Normalize(
                        [0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225])
                    ])
    return transform


def get_dataloaders(args):
    train_dataset = torchvision.datasets.ImageFolder(args.root, transform=get_transforms())

    test_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=args.batch_size,
        num_workers=args.num_workers)
    return test_loader

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
    model.eval()

    # define optimzer and scheduler
    criterion = nn.CrossEntropyLoss()
    # load dataset
    test_loader = get_dataloaders(args)
    # test set
    test_model(
            args,
            model,
            test_loader,
            criterion)
