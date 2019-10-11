import requests


def test_classification():  # 分类
    file_dir = "/Users/lichengzhi/bailian/壳牌/线上数据/error/商丘市何李润滑油有限公司_夏邑叶和正道汽修_刘腾藏_13.jpg"
    data = {"data": "data"}
    file = {"file": open(file_dir, 'rb')}
    # url = "http://100.64.32.2:5001/classification"
    url = "http://bailian-gpu.chinaeast2.cloudapp.chinacloudapi.cn:5001/classification"
    print(requests.post(url=url, files=file).json())


def test_rotation():  # 分类
    file_dir = "/Users/lichengzhi/bailian/壳牌/分类2/倍赛门店及sku标注/门店/广东+汕头市恒越润滑油有限公司+驰铭汽车+苏展/广东+汕头市恒越润滑油有限公司+驰铭汽车+苏展（远景）.jpg"
    data = {"data": "data"}
    file = {"file": open(file_dir, 'rb')}
    # url = "http://100.64.32.2:5001/rotation"
    url = "http://bailian-gpu.chinaeast2.cloudapp.chinacloudapi.cn:5001/rotation"
    print(requests.post(url=url, files=file).json())


def test_detection():       # 检测
    file_dir = "/Users/lichengzhi/bailian/壳牌/线上测试/test20.jpg"
    file = {"file": open(file_dir, 'rb')}
    # url = "http://100.64.32.2:5001/detection"
    url = "http://bailian-gpu.chinaeast2.cloudapp.chinacloudapi.cn:5001/detection"
    print(requests.post(url=url, files=file).json())

