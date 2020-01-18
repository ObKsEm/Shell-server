"""  
Copyright (c) 2019-present NAVER Corp.
MIT License
"""

# -*- coding: utf-8 -*-
import sys
import os
import time
import argparse

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from PIL import Image

import cv2
from skimage import io
import numpy as np
from craft import craft_utils
from craft import imgproc
from craft import file_utils
import json
import zipfile

from craft.craft import CRAFT

from collections import OrderedDict


class CraftModelWrapper:
    def __init__(self,
                 trained_model,
                 text_threshold=0.7,
                 low_text=0.4,
                 link_threshold=0.4,
                 canvas_size=1280,
                 mag_ratio=1.5,
                 poly=False,
                 refine=False,
                 refiner_model=None):

        self.net = CRAFT()  # initialize
        self.text_threshold = text_threshold
        self.low_text = low_text
        self.link_threshold = link_threshold
        self.canvas_size = canvas_size
        self.mag_ratio = mag_ratio
        self.poly = poly
        if torch.cuda.is_available():
            self.net.load_state_dict(self.copyStateDict(torch.load(trained_model)))
            self.net = self.net.cuda()
            self.net = torch.nn.DataParallel(self.net)
            cudnn.benchmark = False
            # self.net.share_memory()
        else:
            self.net.load_state_dict(self.copyStateDict(torch.load(trained_model, map_location='cpu')))
        self.net.eval()
        if refine and refiner_model:

            from craft.refinenet import RefineNet

            self.refine_net = RefineNet()
            if torch.cuda.is_available():
                self.refine_net.load_state_dict(self.copyStateDict(torch.load(refiner_model)))
                refine_net = self.refine_net.cuda()
                refine_net = torch.nn.DataParallel(refine_net)
            else:
                self.refine_net.load_state_dict(self.copyStateDict(torch.load(refiner_model, map_location='cpu')))
            self.refine_net.eval()
            self.poly = True
        else:
            self.refine_net = None

    def copyStateDict(self, state_dict):
        if list(state_dict.keys())[0].startswith("module"):
            start_idx = 1
        else:
            start_idx = 0
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = ".".join(k.split(".")[start_idx:])
            new_state_dict[name] = v
        return new_state_dict

    def str2bool(self, v):
        return v.lower() in ("yes", "y", "true", "t", "1")

    def test_net(self, image):
        t0 = time.time()

        # resize
        img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, self.canvas_size, interpolation=cv2.INTER_LINEAR,
                                                                              mag_ratio=self.mag_ratio)
        ratio_h = ratio_w = 1 / target_ratio

        # preprocessing
        x = imgproc.normalizeMeanVariance(img_resized)
        x = torch.from_numpy(x).permute(2, 0, 1)  # [h, w, c] to [c, h, w]
        x = Variable(x.unsqueeze(0))  # [c, h, w] to [b, c, h, w]
        if torch.cuda.is_available():
            x = x.cuda()

        # forward pass
        with torch.no_grad():
            y, feature = self.net(x)

        # make score and link map
        score_text = y[0, :, :, 0].cpu().data.numpy()
        score_link = y[0, :, :, 1].cpu().data.numpy()

        # refine link
        if self.refine_net is not None:
            y_refiner = self.refine_net(y, feature)
            score_link = y_refiner[0, :, :, 0].cpu().data.numpy()

        t0 = time.time() - t0
        t1 = time.time()

        # Post-processing
        character_boxes, boxes, polys = craft_utils.getDetBoxes(score_text, score_link, self.text_threshold, self.link_threshold, self.low_text, self.poly)

        # coordinate adjustment
        # character_boxes = [craft_utils.adjustResultCoordinates(cbox, ratio_w, ratio_h) for cbox in character_boxes]
        boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
        polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
        for k in range(len(polys)):
            if polys[k] is None:
                polys[k] = boxes[k]

        t1 = time.time() - t1

        # render results (optional)
        render_img = score_text.copy()
        render_img = np.hstack((render_img, score_link))
        ret_score_text = imgproc.cvt2HeatmapImg(render_img)

        print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))

        return polys

    def text_detection(self, image_path):
        image = imgproc.loadImage(image_path)
        polys = self.test_net(image)
        ret_poly = []
        height = image.shape[0]
        width = image.shape[1]
        for i, box in enumerate(polys):
            poly = np.array(box).astype(np.int32).reshape((-1))
            poly = poly.reshape(-1, 2)
            x = poly[:, 0]
            y = poly[:, 1]
            x = x.clip(0, width)
            y = y.clip(0, height)
            ret_poly.append(np.dstack((x, y))[0])
        return ret_poly

