import math
from io import BytesIO
import rembg
import requests

from image_type import *


def remove_background_img(output_size, user_data):
    if user_data['tmp_img'] is None:
        return None

    img = np2pil(user_data['tmp_img'])
    # img = user_data['tmp_img']
    output = rembg.remove(
        img
    )
    # output.show()
    output = pil2np(output)

    output_cv = cv2.cvtColor(output, cv2.COLOR_RGBA2BGRA)
    r, g, b, a = cv2.split(output_cv)

    _, mask = cv2.threshold(a, 127, 255, cv2.THRESH_BINARY)

    contous, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if len(contous) == 0:
        return output

    box_list = []
    area_list = []
    for cont in contous:
        x, y, w, h = cv2.boundingRect(cont)
        box_list.append([x, y, w, h])
        area_list.append(w * h)

    max_box = box_list[np.argmax(area_list)]
    x, y, w, h = max_box
    cx, cy = x + w / 2, y + h / 2
    nw, nh = w * 1.1, h * 1.1
    nx1, ny1 = math.ceil(cx - nw / 2), math.ceil(cy - nh / 2)
    nx2, ny2 = math.ceil(cx + nw / 2), math.ceil(cy + nh / 2)

    cut_img = output_cv[ny1:ny2, nx1:nx2]
    if cut_img.shape[0] > cut_img.shape[1]:
        top = bottom = cut_img.shape[0] // 4
        left = right = (cut_img.shape[0] + top + bottom - cut_img.shape[1]) // 2
    else:
        left = right = cut_img.shape[1] // 4
        top = bottom = (cut_img.shape[1] + left + right - cut_img.shape[0]) // 2

    output_img = cv2.copyMakeBorder(cut_img, top, bottom, left, right, cv2.BORDER_CONSTANT, (0, 0, 0, 0))
    output = cv2.cvtColor(output_img, cv2.COLOR_BGRA2RGBA)

    user_data['output_img'] = output
    return download_img(output_size, user_data)


def download_img(size, user_data):
    try:
        size = int(size)
    except:
        size = 500

    if user_data['output_img'] is None:
        return None

    h, w = user_data['output_img'].shape[:2]
    scale = min(size / h, size / w)

    resh, resw = int(scale * h), int(scale * w)
    res_img = cv2.resize(user_data['output_img'], (resw, resh))
    return res_img


def pil2np(img):
    img = np.array(img)
    return img


def get_img_data(img):
    if img.startswith("http"):
        try:
            req = requests.get(img)
            image = Image.open(BytesIO(req.content))
        except Exception as e:
            raise ValueError("get img url error:{}".format(e))
    else:
        try:
            decoded_data = base64.b64decode(img)
            image = Image.open(BytesIO(decoded_data))
        except Exception as e:
            raise ValueError("get img base64 error:{}".format(e))
    return image
