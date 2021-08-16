import time

import IPython
import cv2
import numpy as np
import mlcvtools
import torch
from torchvision import transforms, models
from torch.autograd import Variable
import torch.nn.functional as F
from PIL import ImageOps
from copy import deepcopy

from utils.models import get_mask_rcnn, get_resnet50
from utils.pipeline import SegmentationModel
from utils.args import classification_args

##### Select camera #####
# cam = cv2.VideoCapture(0)       # First camera on PC


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


CLASS_NAMES = ['WaltherP99', 'Glock34', 'Glock19', 'negatives', 'Glock30',
               'Glock21', 'TanfoglioCG', 'Glock26', 'WaltherPPQ', 'ColtM1911',
               'HecklerKochUSP', 'Glock45', 'Glock17', 'WaltherP22',
               'Jericho941F', 'SigSauerP220']


def denormalize(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    inp = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    inp = np.array(inp *255, dtype=np.uint8)
    return inp


def show(pic):
    return cv2.resize(pic, (448, 448))

def run_object_dection(pic):
    inputs = transforms.ToTensor()(pic).to("cuda")
    inputs = SEGMENTATION_MODEL([inputs])
    permute = [2, 1, 0]
    inputs = inputs[:,permute, :, :]
    torch.cuda.empty_cache()
    return inputs

def run_gradcam(pic):
    grad_cam = GradCam(MODEL)
    feature_image = grad_cam(pic).squeeze(dim=0)
    feature_image = np.transpose(feature_image, (1, 2, 0))
    feature_image = np.array(feature_image * 255, dtype=np.uint8)
    feature_image = cv2.cvtColor(feature_image, cv2.COLOR_BGR2RGB)

    output, features = MODEL(pic)
    output = output.detach().cpu()
    background = np.zeros((448, 448, 3), dtype=np.uint8)
    softmax = torch.nn.functional.softmax(output, 1)
    class_score = list(zip(CLASS_NAMES, softmax.squeeze(0)))
    class_score.sort(key=lambda tup:tup[1], reverse=True)
    for i, (name, score) in enumerate(class_score):
        cv2.putText(background, f'{name}: {round(score.item() ,2)}',(10, 100 + i * 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (255,255,255),thickness=2)
        if i > 1:
            break
    torch.cuda.empty_cache()
    return feature_image, background

def apply_all(pic):
    image_name_tpl = []
    original = deepcopy(pic)
    pic = show(pic)
    image_name_tpl.append((pic, "normal"))
    detected = run_object_dection(original)
    results = denormalize(detected.squeeze(0).detach().cpu())
    results = cv2.resize(results, (448,448))
    image_name_tpl.append((results,"Object Detection"))
    grads, score = run_gradcam(detected)
    image_name_tpl.append((grads, "GradCam"))
    image_name_tpl.append((score, "Classification"))
    return mlcvtools.tile_images(image_name_tpl, cols=2, image_width=None)

if __name__ == "__main__":
    args = classification_args()
    SEGMENTATION_MODEL = SegmentationModel(args, "cuda")
    MODEL = get_resnet50("results/experiment5/resnet.pt")
    MODEL.eval()
    MODEL.to('cuda')
    selected_algorithm = apply_all

    # Mobile phone through WIFI
    cam = cv2.VideoCapture()
    cam.open("http://192.168.1.107:8080/?action=stream")
    fourcc = cv2.VideoWriter_fourcc('F','M','P','4')
    videoWriter = cv2.VideoWriter('demo.avi', fourcc, 2.0, (899,923))

    #########################

    fps = 0
    t1 = time.time()
    running = True
    try:
        most_recent_poller = mlcvtools.AsyncMostRecentResultPoller(cam.read)
        most_recent_poller.start()
        while(True):
            try:
                ret, frame = most_recent_poller.get()
                # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # frame = cv2.resize(frame, (640, 480))

                pic = selected_algorithm(frame)

                #### Show in notebook ####

                # Add FPS (frames per second) to show how fast it is
                mlcvtools.put_text(pic, f"{fps:1.2f} fps", 0, 0)

                # Display image inline in the notebook
                #   Use JPEG because it requires less data, this is important because to update
                #   the image is encoded to base64 en decoded again by the browser, this gets
                #   slow for large images when not using compression like JPEG.
                #   Tune the quality and format to your scenario if desired!
                # mlcvtools.display_image_data(pic, fmt='jpeg', quality=90)
                videoWriter.write(pic)
                cv2.imshow("", pic)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                # Schedule removal of the image on next update
                # This does the trick of animating frame after frame!
                # (but only do this while still running so that we can see the last frame when stopping)
                if running:
                    IPython.display.clear_output(wait=True)
                else:
                    break

                # Update FPS (calculate smooth averaged)
                t2 = time.time()
                fps = (fps * 20 + 1/(t2-t1)) / 21
                t1 = t2
            except KeyboardInterrupt as e:
                running = False
    finally:
        most_recent_poller.stop()
        cam.release()
        videoWriter.release()
        print("Done")


