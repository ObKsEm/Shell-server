import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Dataset

from PIL import Image

import os
import random
import numpy as np
from torch.autograd import Variable

model_dir = "./models/resnet18.pt"

class_to_idx = dict({'rests': 0, 'shelf': 1, 'shop': 2})
idx_to_class = dict(zip(class_to_idx.values(), class_to_idx.keys()))


img_transforms = transforms.Compose([
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])


def image_classify(img):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_tensor = img_transforms(img).float()
    image_tensor = image_tensor.unsqueeze_(0)
    image_tensor.to(device)
    import torchvision.models as models
    model = models.resnet18(pretrained=False).to(device)
    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Linear(in_features=512, out_features=3).to(device)
    model.load_state_dict(torch.load(model_dir))
    input = Variable(image_tensor).to(device)
    output = model(input)
    result = output.data.cpu().numpy().argmax()
    return result