import requests


def test_classification():  # 分类
    file_dir = "/Users/lichengzhi/bailian/壳牌/分类1/人工/车立方2.jpg"
    data = {"data": "data"}
    file = {"file": open(file_dir, 'rb')}
    url = "http://100.64.32.2:5001/classification"
    print(requests.post(url=url, files=file).json())


def test_detection():       # 检测
    file_dir = "/Users/lichengzhi/bailian/壳牌/分类1/人工/车立方1.jpg"
    file = {"file": open(file_dir, 'rb')}
    url = "http://100.64.32.2:5001/detection"
    print(requests.post(url=url, files=file).json())

