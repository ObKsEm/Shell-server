import requests
import cv2
import numpy as np


def test_img_read():
    r = requests.get("https://bailian.blob.core.chinacloudapi.cn/shell/sku/18a938df5e293c6e8d31b5cf84d5b5f7.jpg")
    print(r)
    data = r.content
    print(r.headers)
    np_arr = np.frombuffer(data, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    cv2.imshow("test", image)
    cv2.waitKey(0)

