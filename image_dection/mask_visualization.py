import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms, datasets
import random

from utils.models import get_mask_rcnn


CLASSES = ['__background__', 'gun']


def pipeline_collate(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    target = torch.LongTensor(target)
    return [data, target]


def get_prediction(model, images, threshold=.5):
    """
    Get predictions based on the mask r-cnn model
    and a list of images. threshold is standard 0.5
    returns a list of tuples with masks bboxes and class
    predictions.
    """
    predictions = []
    for img in images:
        pred = model([img])
        predictions.extend(pred)
    annotations = []
    for pred in predictions:
        score = list(pred['scores'].detach().cpu().numpy())
        prediction_idx = [x for x in score if x > threshold]
        if len(prediction_idx) == 0:
            annotations.append(([],[],[]))
            continue
        prediction_idx = np.max(prediction_idx)
        prediction_idx = [score.index(prediction_idx)]
        pred_masks = (pred['masks'] >.5).squeeze(1).detach().cpu().numpy()
        pred_classes = np.array([CLASSES[i] for i in pred['labels']])
        pred_boxes = pred['boxes'].detach().cpu().numpy().astype(np.int)
        masks = pred_masks[prediction_idx]
        boxes = pred_boxes[prediction_idx]
        classes = pred_classes[prediction_idx]
        annotations.append((masks, boxes, classes))
    return annotations


def random_colour_masks(mask):
    """
    returns a colored mask a random color
    """
    colors = [0,255,0]
    random.shuffle(colors)
    r = np.zeros_like(mask).astype(np.uint8)
    g = np.zeros_like(mask).astype(np.uint8)
    b = np.zeros_like(mask).astype(np.uint8)
    r[mask == 1], g[mask == 1], b[mask == 1] = colors
    colored_mask = np.stack([r, g, b], axis=2)
    return colored_mask


def visualize(img, pred, rect_th=32, text_size=24, text_th=4):
    """
    Receives a img with prediction tuple (mask,bbox,classes) and
    visualizes the mask and bounding boxes
    """
    masks, boxes, _ = pred
    for mask, box in zip(masks,boxes):
        rgb_mask = random_colour_masks(mask)
        img = cv2.addWeighted(img, 1, rgb_mask, 0.5, 0)
        cv2.rectangle(img, tuple(box[:2]), tuple(box[2:]),color=(0, 255, 0),
                thickness=rect_th)
        # cv2.putText(img, 'gun', tuple(box[:2]), cv2.FONT_HERSHEY_SIMPLEX,
                # text_size, (0,255,0),thickness=text_th)
    cv2.imshow("guns", img)
    cv2.waitKey(0)

if __name__ == "__main__":
    model, _ = get_mask_rcnn()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load("results/mask_rcnn/mask_rcnn_1615499715.2833326.pt"))
    model.eval()
    model.to(device)
    transform = transforms.Compose([
            transforms.ToTensor(),
                    ])
    dataset = datasets.ImageFolder("../example_images", transform= transform)
    dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            collate_fn= pipeline_collate )
    for images, _ in dataloader:
        images = list(img.to(device) for img in images)
        pred = get_prediction(model, images, threshold=.5)
        for p, i in zip(pred, images):
            i = np.array(i.detach().cpu() * 255, dtype=np.uint8).transpose(1, 2, 0)
            i = cv2.cvtColor(i, cv2.COLOR_BGR2RGB)
            visualize(i, p)

