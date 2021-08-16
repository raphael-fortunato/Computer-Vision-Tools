import os
import numpy as np
import torch
import torchvision
from torchvision import transforms
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.model_selection import train_test_split
import albumentations as A

from image_dection import detection_utils
from utils.custom_datasets import Detection_Dataset, Segmentation_Dataset, ClassificationDataset
import image_dection.transforms as T






# Detection data augmentations
def get_transforms(train):
    transforms = []
    if train:
        transforms.extend([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomSizedBBoxSafeCrop(448, 448),
                A.ShiftScaleRotate(p=.5),
                ])
    return A.Compose(
            transforms,
            bbox_params=A.BboxParams(
                    format='pascal_voc',
                    min_visibility=.5,
                    label_fields=['category_ids']))



# loads test and train segmentation dataloaders 
def get_segmentation_dataloaders(args):
    # use our dataset and defined transformations
    dataset = Segmentation_Dataset(
            args, get_transforms(train=False), mode='train')
    dataset_test = Segmentation_Dataset(
             args, get_transforms(train=False), mode='test')

    # define training and validation data loaders
    pin_memory = True if args.num_workers == 1 else False
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=detection_utils.collate_fn,
        pin_memory=pin_memory)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=detection_utils.collate_fn,
        pin_memory=pin_memory)
    return data_loader, data_loader_test


# load test and train detection dataloaders
def get_detection_dataloaders(args):
    dataset_test = Detection_Dataset(
            args,
            get_transforms(True))
    dataset = Detection_Dataset(
            args,
            get_transforms(True))
    print(f"Dataset Size: {len(dataset)}")
    dataset_len = len(dataset)
    indices = torch.randperm(dataset_len).tolist()
    split = int(dataset_len * .5)
    dataset = torch.utils.data.Subset(dataset, indices[:split])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[split:])
    print(
    f"Length Trainset: {len(dataset)}, Length Testset: {len(dataset_test)}")

    # define training and validation data loaders
    pin_memory = True if args.num_workers == 1 else False
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, num_workers=args.num_workers,
        shuffle=True, collate_fn=detection_utils.collate_fn,
        pin_memory=pin_memory)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size, num_workers=args.num_workers,
        collate_fn=detection_utils.collate_fn, pin_memory=pin_memory)
    return data_loader, data_loader_test
