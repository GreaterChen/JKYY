import numpy as np
from PIL import Image
import cv2
import base64
import io


def pil2cv(img):
    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    return img


def cv2pil(img):
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return img


def np2cv(img, type='rgb'):
    if type == 'rgb':
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if type == 'bgr':
        img = img

    return img


def np2pil(img, type='rgb'):
    if type == 'rgb':
        img = Image.fromarray(img)
    if type == 'bgr':
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    return img


def pil2np(img):
    img = np.array(img)
    return img


def cv2np(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def pil2base64(img):
    image_data = io.BytesIO()
    img.save(image_data, format('JPEG'))
    image_data_bytes = image_data.getvalue()
    encoded_image = base64.b64encode(image_data_bytes).decode('utf-8')
    return encoded_image


def base642pil(img_string):
    init_image = Image.open(io.BytesIO(base64.b64decode(img_string))).convert("RGB")
    return init_image


def cv2base64(img):
    img = cv2.imencode('.jpg', img)[1]
    image_code = str(base64.b64encode(img))[2:-1]

    return image_code

# def base642cv(img_string):
#     img_data = base64.b64decode(base64_code)
#     img_array = np.fromstring(img_data, np.uint8)
#     img = cv2.imdecode(img_array, cv2.COLOR_RGB2BGR)
#
#     return img
