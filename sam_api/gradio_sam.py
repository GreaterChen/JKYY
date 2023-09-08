import os
import math
import cv2
import numpy as np
import gradio as gr
import rembg
import warnings
from image_type import np2pil, pil2np

warnings.filterwarnings("ignore")
IMG_MAX_SIZE = 500


def get_preview(img):
    global IMG_MAX_SIZE
    h, w = img.shape[:2]
    scale = min(IMG_MAX_SIZE / w, IMG_MAX_SIZE / h)
    res_w = int(w * scale)
    res_h = int(h * scale)

    res_img = cv2.resize(img, (res_w, res_h))

    return res_img


def input(img, user_data):
    user_data['origin_img'] = img
    user_data['input_img'] = img
    user_data['tmp_img'] = img
    user_data['output_img'] = img
    user_data['include_points'] = []

    if user_data['input_img'] is None:
        return None

    return get_preview(user_data['input_img'])


def clear(user_data):
    user_data['origin_img'] = None
    user_data['input_img'] = None
    user_data['tmp_img'] = None
    user_data['output_img'] = None
    return None


def reset_img(user_data):
    if user_data['input_img'] is None:
        return None, 0

    user_data['input_img'] = user_data['origin_img']
    user_data['tmp_img'] = user_data['origin_img']

    return get_preview(user_data['input_img']), 0


def flip_img(tmp_img_angle, user_data):
    if user_data['input_img'] is None:
        return None

    img = user_data['input_img'].copy()

    img = cv2.flip(img, 1)
    user_data['tmp_img'] = img

    h, w = img.shape[:2]
    rotate_center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(rotate_center, tmp_img_angle, 1.0)
    new_w = int(h * np.abs(M[0, 1]) + w * np.abs(M[0, 0]))
    new_h = int(h * np.abs(M[0, 0]) + w * np.abs(M[0, 1]))
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2
    user_data['tmp_img'] = cv2.warpAffine(img, M, (new_w, new_h))

    return get_preview(user_data['tmp_img'])


def rotate_img(angle, user_data):
    if user_data['input_img'] is None:
        return None

    img = user_data['input_img'].copy()
    h, w = img.shape[:2]
    rotate_center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(rotate_center, angle, 1.0)
    new_w = int(h * np.abs(M[0, 1]) + w * np.abs(M[0, 0]))
    new_h = int(h * np.abs(M[0, 0]) + w * np.abs(M[0, 1]))
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2

    user_data['tmp_img'] = cv2.warpAffine(img, M, (new_w, new_h))

    return get_preview(user_data['tmp_img'])


def get_point(evt: gr.SelectData, user_data):
    if user_data['input_img'] is None:
        return None
    user_data['include_points'].append(evt.index)

    return user_data['include_points']


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


def remove_background_img(output_size, user_data):
    if user_data['tmp_img'] is None:
        return None

    img = np2pil(user_data['tmp_img'])
    # img = user_data['tmp_img']
    output = rembg.remove(
        img
    )
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


with gr.Blocks() as demo:
    user_data = {"origin_img": None,
                 "input_img": None,
                 "include_points": None,
                 "tmp_img": None,
                 "output_img": None
                 }

    stats = gr.State(user_data)

    with gr.Row():
        with gr.Column():
            with gr.Box():
                with gr.Row():
                    input_img = gr.Image(label='输入图像')
                with gr.Row():
                    statement = gr.Textbox(label="包含的点")
                with gr.Row():
                    with gr.Column():
                        reset = gr.Button(value='重置')
                    with gr.Column():
                        flip = gr.Button(value='镜像翻转')
                with gr.Row():
                    angle = gr.Slider(0, 360, label="旋转")
                with gr.Row():
                    submit = gr.Button(value="上传")
        with gr.Column():
            with gr.Box():
                with gr.Row():
                    output_img = gr.Image(label='输出图像')
                with gr.Row():
                    dw_size = gr.Dropdown(
                        [str(500), str(800), str(1000), str(1500), str(2000)],
                        label="输出尺寸 (n x n), 默认500"
                    )

    # Func >>>>>>>>>>>>>>>>>>>>>>>
    input_img.upload(
        input,
        [input_img, stats],
        input_img
    )

    input_img.clear(
        clear,
        [stats],
        input_img
    )

    input_img.select(
        get_point,
        [stats],
        statement
    )

    reset.click(
        reset_img,
        [stats],
        [input_img, angle]
    )

    flip.click(
        flip_img,
        [angle, stats],
        input_img
    )

    angle.change(
        rotate_img,
        [angle, stats],
        input_img
    )

    submit.click(
        remove_background_img,
        [dw_size, stats],
        output_img
    )

    dw_size.change(
        download_img,
        [dw_size, stats],
        output_img
    )

if __name__ == '__main__':
    demo.queue().launch(share=False, inbrowser=True,
                        server_name="127.0.0.1",
                        server_port=18401
                        # root_path="/HeadView/Web"
                        )
