import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.autograd import Variable
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import ImageOps

from utils.models import get_vgg, get_resnet
from utils.args import classification_args
from utils.dataloaders import get_dataset
from utils.custom_datasets import ClassificationDataset
from utils.pipeline import SegmentationModel


class GradCam:
    def __init__(self, model):
        self.model = model.eval()
        self.feature = None
        self.gradient = None

    def save_gradient(self, grad):
        self.gradient = grad

    def __call__(self, x):
        image_size = (x.size(-1), x.size(-2))
        datas = Variable(x)
        heat_maps = []
        for i in range(datas.size(0)):
            img = datas[i].data.cpu().numpy()
            img = img - np.min(img)
            if np.max(img) != 0:
                img = img / np.max(img)

            feature = datas[i].unsqueeze(0)
            for name, module in self.model.named_children():
                if name == 'dense':
                    feature = feature.view(feature.size(0), -1)
                feature = module(feature)
                if name == 'backbone':
                    feature.register_hook(self.save_gradient)
                    self.feature = feature
            classes = torch.sigmoid(feature)
            one_hot, _ = classes.max(dim=-1)
            self.model.zero_grad()
            one_hot.backward()

            weight = self.gradient.mean(dim=-1, keepdim=True).mean(dim=-2, keepdim=True)
            mask = F.relu((weight * self.feature).sum(dim=1)).squeeze(0)
            mask = cv2.resize(mask.data.cpu().numpy(), image_size)
            mask = mask - np.min(mask)
            if np.max(mask) != 0:
                mask = mask / np.max(mask)
            heat_map = np.float32(cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET))
            cam = heat_map + np.float32((np.uint8(img.transpose((1, 2, 0)) * 255)))
            cam = cam - np.min(cam)
            if np.max(cam) != 0:
                cam = cam / np.max(cam)
            heat_maps.append(transforms.ToTensor()(cv2.cvtColor(np.uint8(255 * cam), cv2.COLOR_BGR2RGB)))
        heat_maps = torch.stack(heat_maps)
        return heat_maps

if __name__ == "__main__":
    args = classification_args()
    for arg in vars(args):
        print(arg, getattr(args, arg))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # setting seeds for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    data1 = get_dataset("../grad_e    fig, axes = plt.subplots(
            nrows=len(experiments),
            ncols=len(data), figsize=(12, 8))
    gs1 = gridspec.GridSpec(3, 6)
    gs1.update(wspace=0.025, hspace=0.05)
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((448, 448)),
            transforms.Normalize(
                    [0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225])])

    dataset = torchvision.datasets.ImageFolder(
            args.root,
            transform=transforms
            )
    class_labels = dataset.targets
    dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            shuffle=True,
            batch_size=1
            )
    # axes[i, 0].set_ylabel(ex, rotation=90, fontsize=20)
    model = get_resnet(args.path)
    model.eval()
    model.to(device)

    grad_cam = GradCam(model)
    images = []
    labels = []
    predictions = []
    for idx, (inputs, label) in enumerate(dataloader):
        inputs = inputs.to(device)
        feature_image = grad_cam(inputs).squeeze(dim=0)
        feature_image = transforms.ToPILImage()(feature_image)
        outputs, features = model(inputs)
        _, preds = torch.max(outputs, 1)
        preds = preds.cpu()
        images.append(feature_image)
        predictions.append(preds)
        labels.append(label)

    for j, (img, label, pred) in enumerate(zip(images, labels, predictions)):
        if i == 0:
            axes[i, j].set_title(class_labels[label], fontsize=20)
        if label == pred:
            img = ImageOps.expand(img, border=15, fill='green')
        else:
            img = ImageOps.expand(img, border=15, fill='red')
        axes[i, j].grid(False)
        axes[i, j].set_xticks([])
        axes[i, j].set_yticks([])
        axes[i, j].imshow(img)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
