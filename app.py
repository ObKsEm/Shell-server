# -*- coding: utf-8 -*-
import multiprocessing
import os
import re
import traceback
from collections import defaultdict

from sanic import Sanic
from sanic.exceptions import NotFound
from sanic.log import logger
from sanic.response import json
# from sanic_openapi import swagger_blueprint, doc
from classifier import image_classify

from PIL import Image
from io import BytesIO


app = Sanic("Shell Api", strict_slashes=True)
app.blueprint(swagger_blueprint)
app.config.API_TITLE = 'Shell API'
app.config.API_DESCRIPTION = 'A Toolkit for image classification and object detector'
app.config.API_PRODUCES_CONTENT_TYPES = ['application/json']

RE_BACKSPACES = re.compile("\b+")

model_name = os.environ.get("MODEL_NAME", 'news').lower()
n_workers = int(os.environ.get('WORKERS', multiprocessing.cpu_count()))

# model_dir = f"/familia/model/{model_name}"
model_dir = f"/Users/lichengzhi/bailian/workspace/Familia/model/{model_name}"
emb_file = f"{model_name}_twe_lda.model"


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
    data = {'success': success, 'message': message, 'data': data}
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


@app.route('/classification', methods=["POST"])
async def api_classification(request):
    try:
        test_file = request.files.get('file')
        file_parameters = {
            'body': test_file.body,
            'name': test_file.name,
            'type': test_file.type,
        }
        if file_parameters["body"] is None:
            return error_response()
        image = Image.open(BytesIO(file_parameters["body"]))
        result = image_classify(image)
        return response(data=result)
    except Exception as err:
        logger.error(err, exc_info=True)
        return error_response(str(err))


if __name__ == '__main__':
    logger.info(f"running shell api with {n_workers} workers")
    app.run(host='0.0.0.0', port=5001, workers=1)
