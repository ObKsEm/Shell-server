# -*- coding: utf-8 -*-
import multiprocessing
import os
import re
import traceback
from collections import defaultdict, Counter

from sanic import Sanic
from sanic.exceptions import NotFound
from sanic.log import logger
from sanic.response import json
from sanic_openapi import swagger_blueprint, doc

from craft.craft_wrapper import CraftModelWrapper
from craft.util.request_for_ocr import request_for_ocr
from model_wrapper import DetectorModelWrapper, RotatorModelWrapper, DoubleDetectorModelWrapper, KvModelWrapper
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
import cv2
import numpy as np

from mmdet.datasets.Ultra4 import Ultra4Dataset, Ultra4MidDataset, Ultra4SimplifiedMidDataset, Ultra4SimplifiedDataset
from mmdet.datasets.crto import CRTODataset, CRTOMidDataset

app = Sanic("Shell Api", strict_slashes=True)
app.blueprint(swagger_blueprint)
app.config.API_TITLE = 'Shell API'
app.config.API_DESCRIPTION = 'A Toolkit for image classification and object detector'
app.config.API_PRODUCES_CONTENT_TYPES = ['application/json']

RE_BACKSPACES = re.compile("\b+")

cls_model_name = os.environ.get("CLS_MODEL_NAME", 'classifier').lower()
det_model_name = os.environ.get("DET_MODEL_NAME", 'detector').lower()
detail_det_model_name = os.environ.get("DETAIL_DET_MODEL_NAME", 'detail_detector').lower()
rot_model_name = os.environ.get("ROT_MODEL_NAME", 'rotator').lower()
ocr_model_name = os.environ.get("OCR_MODEL_NAME", 'craft').lower()
rg_det_model_name = os.environ.get("RG_DET_MODEL_NAME", "rg_detector").lower()
ab_det_model_name = os.environ.get("AB_DET_MODEL_NAME", "ab_detector").lower()
kv_model_name = os.environ.get("KV_MODEL_NAME", 'kv_detector').lower()
crto_det_model_name = os.environ.get("CRTO_DET_MODEL_NAME", 'crto_detector').lower()

# n_workers = int(os.environ.get('WORKERS', multiprocessing.cpu_count()))
n_workers = 1

cls_model_dir = f"./models/{cls_model_name}"
det_model_dir = f"./models/{det_model_name}"
crto_det_model_dir= f"./models/{crto_det_model_name}"
detail_det_model_dir = f"./models/{detail_det_model_name}"
rot_model_dir = f"./models/{rot_model_name}"
ocr_model_dir = f"./models/{ocr_model_name}"
rg_det_model_dir = f"./models/{rg_det_model_name}"
ab_det_model_dir = f"./models/{ab_det_model_name}"
kv_model_dir = f"./models/{kv_model_name}"


det_config_dir = f"./configs/{det_model_name}.py"
detail_det_config_dir = f"./configs/{detail_det_model_name}.py"
crto_det_config_dr = f"./configs/{crto_det_model_name}.py"
rg_det_config_dir = f"./configs/{rg_det_model_name}.py"
ab_det_config_dir = f"./configs/{ab_det_model_name}.py"
kv_config_dir = f"./configs/{kv_model_name}.py"


# cls_model = ClassifierModelWrapper(cls_model_dir)
det_model = DetectorModelWrapper(det_model_dir, det_config_dir, Ultra4SimplifiedDataset.CLASSES, Ultra4SimplifiedMidDataset.CLASSES)
detail_det_model = DetectorModelWrapper(detail_det_model_dir, detail_det_config_dir, Ultra4Dataset.CLASSES, Ultra4MidDataset.CLASSES)
crto_det_model = DetectorModelWrapper(crto_det_model_dir, crto_det_config_dr, CRTODataset.CLASSES, CRTOMidDataset.CLASSES)
kv_model = KvModelWrapper(kv_model_dir, kv_config_dir)
# det_model = DoubleDetectorModelWrapper(rg_det_model_dir, rg_det_config_dir, ab_det_model_dir, ab_det_config_dir)
rot_model = RotatorModelWrapper(rot_model_dir)
ocr_model = CraftModelWrapper(ocr_model_dir)


def get_apperence(word, sentence):
    ret = 0
    head = 0
    tail = len(sentence)
    pos = sentence.find(word, head, tail)
    while pos > 0:
        ret += 1
        head = pos + len(word)
        pos = sentence.find(word, head, tail)
    return ret


def check_repeat(word, tags):
    x = word.lower()
    for item in tags:
        y = item.lower()
        if y.find(x) > -1 or x.find(y) > -1:
            return True
    return False


def get_param(request, param_name, default_value=None, is_list=False):
    param_value = (request.form.getlist(param_name) if is_list else request.form.getlist(param_name)) or \
                  request.args.get(param_name) or \
                  default_value
    if param_value is None:
        return param_value
    value_type = type(param_value)
    if is_list:
        return param_value if value_type == list else [param_value]
    return param_value[0] if value_type == list else param_value


def strip_to_none(text: str):
    if text is None:
        return None
    text = text.strip()
    text = re.sub(RE_BACKSPACES, '', text)
    if len(text) == 0:
        return None
    if text == 'None':
        return None
    return text


def response(success: bool = True, data=None, message=None):
    if success:
        code = 200
    else:
        code = 500
    data = {'code': code, 'message': message, 'data': data}
    data = {k: v for k, v in data.items() if v is not None}
    try:
        return json(data, ensure_ascii=False)
    except Exception as err:
        logger.error(err, exc_info=True)
        msg = traceback.format_exc()
        data = {'success': success, 'message': msg}
        return json(data, ensure_ascii=False)


def error_response(message='Invalid request'):
    return response(success=False, message=message)


def handle_404(request, exception):
    return api_index(request)


def handle_exception(request, exception):
    return error_response(str(exception))


@app.route('/')
@doc.description("ping")
def api_index(request):
    message = f"Shell API is running, check out the api doc at http://{request.host}/swagger/"
    return response(message=message)


@app.exception(NotFound)
async def ignore_404s(request, exception):
    message = f"Yep, I totally found the page: {request.url}"
    return response(message=message)


# @app.route('/classification', methods=["POST"])
# async def api_classification(request):
#     try:
#         data_file = request.files.get('file')
#         if data_file is None:
#             return error_response("Request for none file data")
#         file_parameters = {
#             'body': data_file.body,
#             'name': data_file.name,
#             'type': data_file.type,
#         }
#         if file_parameters["body"] is None:
#             return error_response("None file body")
#         image = Image.open(BytesIO(file_parameters["body"]))
#         if image.mode is not "RGB":
#             image = image.convert("RGB")
#         softmax, result = cls_model.classify(image)
#         logger.info(f"Classification softmax: {softmax}, result: {result}")
#         return response(data=result)
#     except Exception as err:
#         logger.error(err, exc_info=True)
#         return error_response(str(err))


@app.route('/rotation', methods=["POST"])
async def api_rotation(request):
    try:
        data_file = request.files.get('file')
        if data_file is None:
            return error_response("Request for none file data")
        file_parameters = {
            'body': data_file.body,
            'name': data_file.name,
            'type': data_file.type,
        }
        if file_parameters["body"] is None:
            return error_response("None file body")
        image = Image.open(BytesIO(file_parameters["body"]))
        if image.mode is not "RGB":
            image = image.convert("RGB")
        softmax, result = rot_model.rotate(image)
        logger.info(f"Rotation softmax: {softmax}, result: {result}")
        return response(data=result)
    except Exception as err:
        logger.error(err, exc_info=True)
        return error_response(str(err))


@app.route('/detection', methods=["POST"])
async def api_detection(request):
    try:
        data_file = request.files.get('file')
        if data_file is None:
            return error_response("Request for none file data")
        file_parameters = {
            'body': data_file.body,
            'name': data_file.name,
            'type': data_file.type,
        }
        if file_parameters["body"] is None:
            return error_response("None file body")
        np_arr = np.frombuffer(file_parameters["body"], np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        bboxes, labels = det_model.detect(image)
        logger.info(f'Detection result bboxes: {bboxes}')
        logger.info(f'Detection result labels: {labels}')
        counters = Counter(labels)
        return response(data={"qualified": 1, "sku": counters, "bboxes": bboxes, "labels": labels})
    except Exception as err:
        logger.error(err, exc_info=True)
        return error_response(str(err))


@app.route('/detection_crto', methods=["POST"])
async def api_detection(request):
    try:
        data_file = request.files.get('file')
        if data_file is None:
            return error_response("Request for none file data")
        file_parameters = {
            'body': data_file.body,
            'name': data_file.name,
            'type': data_file.type,
        }
        if file_parameters["body"] is None:
            return error_response("None file body")
        np_arr = np.frombuffer(file_parameters["body"], np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        bboxes, labels = crto_det_model.detect(image)
        logger.info(f'Detection result bboxes: {bboxes}')
        logger.info(f'Detection result labels: {labels}')
        counters = Counter(labels)
        return response(data={"qualified": 1, "sku": counters, "bboxes": bboxes, "labels": labels})
    except Exception as err:
        logger.error(err, exc_info=True)


@app.route('/detection_detail', methods=["POST"])
async def api_detection(request):
    try:
        data_file = request.files.get('file')
        if data_file is None:
            return error_response("Request for none file data")
        file_parameters = {
            'body': data_file.body,
            'name': data_file.name,
            'type': data_file.type,
        }
        if file_parameters["body"] is None:
            return error_response("None file body")
        np_arr = np.frombuffer(file_parameters["body"], np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        bboxes, labels = detail_det_model.detect(image)
        logger.info(f'Detection result bboxes: {bboxes}')
        logger.info(f'Detection result labels: {labels}')
        counters = Counter(labels)
        return response(data={"qualified": 1, "sku": counters, "bboxes": bboxes, "labels": labels})
    except Exception as err:
        logger.error(err, exc_info=True)
        return error_response(str(err))


@app.route('/detection_crto_name', methods=["POST"])
async def api_detection(request):
    try:
        data_file = request.files.get('file')
        if data_file is None:
            return error_response("Request for none file data")
        file_parameters = {
            'body': data_file.body,
            'name': data_file.name,
            'type': data_file.type,
        }
        if file_parameters["body"] is None:
            return error_response("None file body")
        np_arr = np.frombuffer(file_parameters["body"], np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        bboxes, labels = crto_det_model.detect_name(image)
        logger.info(f'Detection result bboxes: {bboxes}')
        logger.info(f'Detection result labels: {labels}')
        counters = Counter(labels)
        return response(data={"qualified": 1, "sku": counters, "bboxes": bboxes, "labels": labels})
    except Exception as err:
        logger.error(err, exc_info=True)
        return error_response(str(err))


@app.route('/detection_kv', methods=["POST"])
async def api_detection_kv(request):
    try:
        data_file = request.files.get('file')
        if data_file is None:
            return error_response("Request for none file data")
        file_parameters = {
            'body': data_file.body,
            'name': data_file.name,
            'type': data_file.type,
        }
        if file_parameters["body"] is None:
            return error_response("None file body")
        np_arr = np.frombuffer(file_parameters["body"], np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        bboxes, labels = kv_model.detect(image)
        logger.info(f'Detection result bboxes: {bboxes}')
        logger.info(f'Detection result labels: {labels}')
        # counters = Counter(labels)
        if len(bboxes) > 0:
            ret = 1
        else:
            ret = 0
        return response(data={"qualified": 1, "kv": ret})
    except Exception as err:
        logger.error(err, exc_info=True)
        return error_response(str(err))


@app.route('/detection_name', methods=["POST"])
async def api_detection_name(request):
    try:
        data_file = request.files.get('file')
        if data_file is None:
            return error_response("Request for none file data")
        file_parameters = {
            'body': data_file.body,
            'name': data_file.name,
            'type': data_file.type,
        }
        if file_parameters["body"] is None:
            return error_response("None file body")
        np_arr = np.frombuffer(file_parameters["body"], np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        bboxes, labels = det_model.detect_name(image)
        logger.info(f'Detection result bboxes: {bboxes}')
        logger.info(f'Detection result labels: {labels}')
        counters = Counter(labels)
        return response(data={"qualified": 1, "sku": counters, "bboxes": bboxes, "labels": labels})
    except Exception as err:
        logger.error(err, exc_info=True)
        return error_response(str(err))


@app.route('/detection_detail_name', methods=["POST"])
async def api_detection_name(request):
    try:
        data_file = request.files.get('file')
        if data_file is None:
            return error_response("Request for none file data")
        file_parameters = {
            'body': data_file.body,
            'name': data_file.name,
            'type': data_file.type,
        }
        if file_parameters["body"] is None:
            return error_response("None file body")
        np_arr = np.frombuffer(file_parameters["body"], np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        bboxes, labels = detail_det_model.detect_name(image)
        logger.info(f'Detection result bboxes: {bboxes}')
        logger.info(f'Detection result labels: {labels}')
        counters = Counter(labels)
        return response(data={"qualified": 1, "sku": counters, "bboxes": bboxes, "labels": labels})
    except Exception as err:
        logger.error(err, exc_info=True)
        return error_response(str(err))


@app.route('/ocr', methods=["POST"])
async def api_ocr(request):
    try:
        data_file = request.files.get('file')
        if data_file is None:
            return error_response("Request for n:one file data")
        file_parameters = {
            'body': data_file.body,
            'name': data_file.name,
            'type': data_file.type,
        }
        if file_parameters["body"] is None:
            return error_response("None file body")
        np_arr = np.frombuffer(file_parameters["body"], np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        poly = ocr_model.text_detection(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        # for i, poly in enumerate(polys):
        x = poly[:, 0]
        y = poly[:, 1]
        xmin = np.min(x)
        ymin = np.min(y)
        xmax = np.max(x)
        ymax = np.max(y)
        poly_img = image[ymin: ymax, xmin: xmax]
        result = request_for_ocr(poly_img)
        text = result["data"]["text"]
        confidence_list = result["data"]["prob"]
        logger.info(f"Request for ocr server: {result}")
        ret = True
        if len(confidence_list) < 1:
            logger.info(f"No character recognized.")
            ret = False
        # conf_list.append(conf)
        if ret:
            return response(data={"qualified": 1, "polygon": poly.tolist(), "text": text})
        else:
            return response(data={"qualified": 1, "polygon": [], "text": ""})
    except Exception as err:
        logger.error(err, exc_info=True)
        return error_response(str(err))


if __name__ == '__main__':
    logger.info(f"running shell api with {n_workers} workers")
    app.run(host='0.0.0.0', port=5001, workers=1)
