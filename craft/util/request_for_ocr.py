import requests
from requests.adapters import HTTPAdapter
from requests import Session, exceptions
import cv2
import numpy as np


def request_for_ocr(image):
    s = Session()
    s.mount("http://", HTTPAdapter(max_retries=5))
    s.mount("https://", HTTPAdapter(max_retries=5))
    img_encode = cv2.imencode('.jpg', image)[1]
    data_encode = np.array(img_encode)
    str_encode = data_encode.tostring()
    file = {"file": ("ocr.jpg", str_encode)}
    url = "http://0.0.0.0:51002/ocr-line"
    try:
        result = s.post(url=url, files=file).json()
    except Exception as e:
        result = {"code": -1, "message": e}
    return result
