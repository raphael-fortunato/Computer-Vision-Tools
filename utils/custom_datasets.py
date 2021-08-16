import os
import glob
import cv2
import numpy as np
import torch
from PIL import Image
import xml.etree.ElementTree as ET
import pandas as pd

from torchvision.transforms import functional as F


# Segmentation Dataset loads images and labels
class Segmentation_Dataset(object):
    def __init__(self, args, transforms, mode='train'):
        self.mode = mode
        self.transforms = transforms
        self.root = args.seg_root
        self.imgs = \
        list(sorted(os.listdir(
            os.path.join(self.root, 'images', mode))))
        self.masks = list(sorted(os.listdir(
            os.path.join(self.root, 'labels', mode))))

    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.root, 'images', self.mode,
                self.imgs[idx])
        mask_path = os.path.join(self.root, 'labels', self.mode,
                self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        img = np.array(img)
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path)

        mask = np.array(mask, dtype=np.uint8)
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)

        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        if self.transforms is not None:
            try:
                transformed = self.transforms(
                        image=img,
                        mask=masks.astype(np.float32).squeeze(0),
                        bboxes=boxes,
                        category_ids=labels)
                transformed_img = transformed['image']
                transformed_boxes = transformed['bboxes']
            except Exception as e:
                print(e)
                print(img_path)
                transformed_boxes = []
        if len(transformed_boxes) > 0:
            img = transformed['image']
            boxes = transformed['bboxes']
            masks = transformed['mask']

        img = F.to_tensor(img)
        masks = torch.tensor(
                np.array(masks).copy(),
                dtype=torch.uint8).unsqueeze(0)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        if boxes.ndim == 1:
            boxes = boxes.unsqueeze(0)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])


        image_id = torch.tensor([idx])

        # suppose all instances are not crowd
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        return img, target

    def __len__(self):
        return len(self.imgs)


# Detection Dataset loads images and labels
class Detection_Dataset(object):
    def __init__(self, args, transforms):
        self.transforms = transforms
        self.labels = \
            list(sorted(os.listdir(
                os.path.join(args.det_root, 'Annotations'))))
        self.imgs = []
        os.chdir(os.path.join(root, 'Images'))
        for label in self.labels[:]:
            id_tag = label[: label.find('.')]
            files = glob.glob(f"{id_tag}*")
            if len(files) != 1:
                self.labels.remove(label)
                continue
            full_path = os.path.join(
                    root,
                    'Images',
                    files[0])
            self.imgs.append(full_path)
        self.labels = [os.path.join(
                        root
                        , 'Annotations',
                        l) for l in self.labels]
        os.chdir("../../../../code")
        assert len(self.imgs) == len(self.labels)

    def __len__(self):
        assert len(self.imgs) == len(self.labels)
        return len(self.imgs)

    def __getitem__(self, idx):
        # load images ad masks
        img_path = self.imgs[idx]
        ann_path = self.labels[idx]

        img = Image.open(img_path)

        # get bounding box coordinates for each mask
        boxes = self.load_bbx(ann_path)
        num_objs = len(boxes)
        img = img.convert("RGB")
        img = np.array(img)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)

        if self.transforms is not None:
            try:
                transformed = self.transforms(
                        image=img,
                        bboxes=boxes,
                        category_ids=labels)
                transformed_img = transformed['image']
                transformed_boxes = transformed['bboxes']
            except Exception as e:
                print(e)
                print(img_path)
                transformed_boxes = []
        if len(transformed_boxes) > 0:
            img = transformed['image']
            boxes = transformed['bboxes']

        img = F.to_tensor(img)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        if boxes.ndim == 1:
            boxes = boxes.unsqueeze(0)

        image_id = torch.tensor([idx])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        return img, target

    def resize_bbx(self, bboxes, scale):
        bbx = []
        for box in bboxes:
            xmin = int(np.round(box[0] * scale[1]))
            xmax = int(np.round(box[2] * scale[1]))
            ymin = int(np.round(box[1] * scale[0]))
            ymax = int(np.round(box[3] * scale[0]))
            assert xmin >= 0 and ymin >= 0
            assert xmin < xmax and ymin < ymax
            bbx.append([xmin, ymin, xmax, ymax])
        return bbx

    def load_bbx(self, filename):
        tree = ET.parse(filename)
        root = tree.getroot()
        boxes = []
        for obj in root.findall('object'):
            bndbox = obj.find('bndbox')
            xmin = int(bndbox.findtext('xmin'))
            ymin = int(bndbox.findtext('ymin'))
            xmax = int(bndbox.findtext('xmax'))
            ymax = int(bndbox.findtext('ymax'))
            assert xmin >= 0 and ymin >= 0
            assert xmin < xmax and ymin < ymax
            boxes.append([xmin, ymin, xmax, ymax])
        return boxes
