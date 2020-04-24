import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from mmdet.ops import nms
import numpy as np
import mmcv
from mmdet.apis import init_detector, inference_detector
from torch.autograd import Variable
import torchvision.models as models
from mmdet.datasets.abrg import ABRGDataset

cls_class_to_idx = dict({'rests': 0, 'shelf': 1, 'shop': 2})
cls_idx_to_class = dict(zip(cls_class_to_idx.values(), cls_class_to_idx.keys()))

rot_class_to_idx = dict({'0': 0, '180': 1, '270': 2, '90': 3})
rot_idx_to_class = dict(zip(rot_class_to_idx.values(), rot_class_to_idx.keys()))


img_transforms = transforms.Compose([
        # transforms.CenterCrop((224, 224)),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])


class ClassifierModelWrapper:
    def __init__(self, model_dir):
        self.model = models.resnet101(pretrained=False)
        for param in self.model.parameters():
            param.requires_grad = False
        fc_features = self.model.fc.in_features
        self.model.fc = nn.Linear(fc_features, out_features=3)
        self.model.load_state_dict(torch.load(model_dir))
        self.model.eval()

    def classify(self, img):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if img.mode is not "RGB":
            img = img.convert("RGB")
        image_tensor = img_transforms(img).float()
        image_tensor = image_tensor.unsqueeze_(0)
        image_tensor.to(device)

        input = Variable(image_tensor).to(device)
        with torch.no_grad():
            output = self.model.to(device)(input)
        softmax = F.softmax(output).cpu().numpy()[0]
        result = output.data.cpu().numpy().argmax()
        return softmax, cls_idx_to_class[result]


class DetectorModelWrapper:
    def __init__(self, model_dir, config_dir):
        cfg = mmcv.Config.fromfile(config_dir)
        cfg.data.test.test_mode = True
        self.model = init_detector(config_dir, model_dir)
        self.model.CLASSES = ABRGDataset.CLASSES

    def detect(self, img):
        result = inference_detector(self.model, img)
        return self.calc_result(result, self.model.CLASSES, score_thr=0.5)

    def calc_result(self, result, class_names, score_thr=0.5):
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
        # ret_bboxes = [bboxes[i].tolist() for i in nms_bboxes[1]]
        ret_bboxes = np.array([bboxes[i] for i in nms_bboxes[1]])
        ret_labels = [class_names[labels[i]] for i in nms_bboxes[1]]
        if len(ret_bboxes) > 0:
            coords = ret_bboxes[:, :4].tolist()
            confidence = ret_bboxes[:, 4].tolist()
        else:
            coords = []
            confidence = []
        return coords, ret_labels


class RotatorModelWrapper:
    def __init__(self, model_dir):
        self.model = models.resnet101(pretrained=False)
        for param in self.model.parameters():
            param.requires_grad = False
        fc_features = self.model.fc.in_features
        self.model.fc = nn.Linear(fc_features, out_features=4)
        self.model.load_state_dict(torch.load(model_dir))
        self.model.eval()

    def rotate(self, img):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if img.mode is not "RGB":
            img = img.convert("RGB")
        image_tensor = img_transforms(img).float()
        image_tensor = image_tensor.unsqueeze_(0)
        image_tensor.to(device)

        input = Variable(image_tensor).to(device)
        with torch.no_grad():
            output = self.model.to(device)(input)
        softmax = F.softmax(output).cpu().numpy()[0]
        result = output.data.cpu().numpy().argmax()
        return softmax, rot_idx_to_class[result]
