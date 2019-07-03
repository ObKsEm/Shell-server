import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Dataset

from PIL import Image
from mmdet.ops import nms
import os
import random
import numpy as np
import mmcv
from mmdet.apis import init_detector, inference_detector
from torch.autograd import Variable
import torchvision.models as models
from mmdet.datasets.shell import ShellDataset

class_to_idx = dict({'rests': 0, 'shelf': 1, 'shop': 2})
idx_to_class = dict(zip(class_to_idx.values(), class_to_idx.keys()))


img_transforms = transforms.Compose([
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])


class ClassifierModelWrapper:
    def __init__(self, model_dir):
        self.model = models.resnet50(pretrained=False)
        for param in self.model.parameters():
            param.requires_grad = False
        fc_features = self.model.fc.in_features
        self.model.fc = nn.Linear(fc_features, out_features=3)
        self.model.load_state_dict(torch.load(model_dir))
        self.model.eval()

    def classify(self, img):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        image_tensor = img_transforms(img).float()
        image_tensor = image_tensor.unsqueeze_(0)
        image_tensor.to(device)

        input = Variable(image_tensor).to(device)
        with torch.no_grad():
            output = self.model.to(device)(input)
        result = output.data.cpu().numpy().argmax()
        return idx_to_class[result]


class DetectorModelWrapper:
    def __init__(self, model_dir, config_dir):
        cfg = mmcv.Config.fromfile(config_dir)
        cfg.data.test.test_mode = True
        self.model = init_detector(config_dir, model_dir)
        self.model.CLASSES = ShellDataset.CLASSES

    def detect(self, img):
        result = inference_detector(self.model, img)
        return self.calc_result(result, self.model.CLASSES, score_thr=0.3)

    def calc_result(self, result, class_names, score_thr=0.3):
        bboxes = np.vstack(result)
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32) for i, bbox in enumerate(result)
        ]
        labels = np.concatenate(labels)
        if score_thr > 0:
            assert bboxes.shape[1] == 5
            scores = bboxes[:, -1]
            inds = scores > score_thr
            bboxes = bboxes[inds, :]
            labels = labels[inds]

        nms_bboxes = nms(bboxes, 0.5)
        ret_bboxes = [bboxes[i].tolist() for i in nms_bboxes[1]]
        ret_labels = [class_names[labels[i]] for i in nms_bboxes[1]]
        return ret_bboxes, ret_labels
