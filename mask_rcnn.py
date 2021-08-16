import os
import numpy as np
import time
import matplotlib
import torch
import torchvision
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from PIL import Image
import xml.etree.ElementTree as ET
from copy import deepcopy

from image_dection.engine import train_one_epoch, evaluate
import image_dection.detection_utils
import image_dection.transforms as T
from utils.models import get_mask_rcnn
from utils.dataloaders import get_detection_dataloaders, get_segmentation_dataloaders
from utils.args import detection_args


def main():
    args = detection_args()
    for arg in vars(args):
        print(arg, getattr(args, arg))

    # setting seeds for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    batch_size = args.batch_size
    num_epochs = args.epochs
    num_workers = args.num_workers
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() \
            else torch.device('cpu')

    # get the model using our helper function
    model, in_features_mask = get_mask_rcnn(args.path)
    model.to(device)

    if args.train_bbx:
        # freezing the mask branch
        mask_branch = deepcopy(model.roi_heads.mask_predictor.state_dict())
        model.roi_heads.mask_predictor = None
        model.__class__ = torchvision.models.detection.faster_rcnn.FasterRCNN

        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(
                params,
                lr=args.lr,
                momentum=args.momentum,
                weight_decay=args.weight_decay)
        # and a learning rate scheduler
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=args.step_size,
                gamma=args.gamma)

        data_loader, data_loader_test = get_detection_dataloaders(args)

        for epoch in range(num_epochs):
            # train for one epoch, printing every 10 iterations
            train_one_epoch(model, optimizer, data_loader, device,
                            epoch, print_freq=10)
            # update the learning rate
            lr_scheduler.step()
            # evaluate on the test dataset
            coco_evaluator = evaluate(model, data_loader_test, device=device)

        # unfreezing the mask branch
        model.__class__ = torchvision.models.detection.mask_rcnn.MaskRCNN
        hidden_layer = 256
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                           hidden_layer,
                                                           2)
        model.roi_heads.mask_predictor.to(device)
        model.roi_heads.mask_predictor.load_state_dict(mask_branch)
        # split mask parameters from rest of the network
        mask_names = ['mask_head', 'mask_predictor']
        mask_params = list(
                map(lambda x: x[1],
                list(
                filter(
                lambda kv: any(param in kv[0] for param in mask_names),
                model.named_parameters()))))
        base_params = list(
                map(lambda x: x[1],
                list(
                filter(
                lambda kv: all(param not in kv[0] for param in mask_names),
                model.named_parameters()))))
        # decrease learning rate for rest of network except mask params
        optimizer = torch.optim.SGD([
                {'params': mask_params, 'lr': args.lr},
                {'params': base_params, 'lr': .0001},
                    ],
                momentum=args.momentum,
                weight_decay=args.weight_decay)
        print(optimizer.param_groups[0]['lr'])
        print(optimizer.param_groups[1]['lr'])

    data_loader, data_loader_test = get_segmentation_dataloaders(args)

    if args.train_segmentation:
        # construct a learning rate scheduler
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=args.step_size,
                gamma=args.gamma)
        for epoch in range(num_epochs):
            # train for one epoch, printing every 10 iterations
            train_one_epoch(model, optimizer, data_loader, device,
                            epoch, print_freq=10)
            # update the learning rate
            lr_scheduler.step()
            # evaluate on the test dataset
            coco_evaluator = evaluate(model, data_loader_test, device=device)

    if args.train_segmentation or args.train_bbx:
        torch.save(
                model.state_dict(),
                f'models/mask_rcnn_{time.time()}.pt'
                )
    model.eval()
    coco_evaluator = evaluate(model, data_loader_test, device=device)
    print("That's it!")

if __name__ == "__main__":
    main()
