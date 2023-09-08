import base64
import json

import requests

url = "http://localhost:8001/upload"  # 接口 URL

# 读取图片文件
with open('../input/truck.jpg', 'rb') as file:
    pic_base64 = base64.b64encode(file.read())
    files = {'img': str(pic_base64, encoding='utf-8'),
             "size": 500,
             "point": "(900,600)"}
    response = requests.post(url, json.dumps(files))

# 检查响应状态码
if response.status_code == 200:
    # 保存响应内容到本地图片文件
    print(response.content)
else:
    print("请求失败")