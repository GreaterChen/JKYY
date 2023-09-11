import os
import math
import cv2
import numpy as np
import gradio as gr
import rembg
import warnings
from image_type import np2pil, pil2np, base642pil, pil2base64
from sam_api.utils import remove_background_img_sam

warnings.filterwarnings("ignore")
IMG_MAX_SIZE = 500


def get_preview(img):
    global IMG_MAX_SIZE
    h, w = img.shape[:2]
    scale = min(IMG_MAX_SIZE / w, IMG_MAX_SIZE / h)
    res_w = int(w * scale)
    res_h = int(h * scale)

    # res_img = cv2.resize(img, (res_w, res_h))
    res_img = img
    return res_img


def input(img, user_data):
    user_data['origin_img'] = img
    user_data['input_img'] = img
    user_data['tmp_img'] = img
    user_data['output_img'] = img
    user_data['include_points'] = []
    user_data['exclude_points'] = []
    user_data['include_area'] = []
    user_data['size'] = IMG_MAX_SIZE
    if user_data['input_img'] is None:
        return None

    return get_preview(user_data['input_img'])


def clear(user_data):
    user_data['origin_img'] = None
    user_data['input_img'] = None
    user_data['tmp_img'] = None
    user_data['output_img'] = None
    user_data['include_points'] = None
    user_data['exclude_points'] = None
    user_data['include_area'] = None
    user_data['size'] = 500
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

    img = user_data['input_img'].copy()

    user_data['include_points'].append(evt.index)

    return user_data['include_points']


def download_img(size, user_data):
    try:
        size = int(size)
    except:
        size = 500

    if user_data['output_img'] is None:
        return None

    imgs = user_data['output_img']
    h, w = imgs[0].shape[:2]
    scale = min(size / h, size / w)
    resh, resw = int(scale * h), int(scale * w)

    for i in range(len(imgs)):
        imgs[i] = cv2.resize(imgs[i], (resw, resh))

    return imgs[-1], imgs[0], imgs[1], imgs[2]


def remove_background_img(user_data):
    if user_data['tmp_img'] is None:
        return None
    output, scores = remove_background_img_sam(
        user_data['size'],
        pil2base64(np2pil(user_data['tmp_img'])),
        user_data['include_points'],
        user_data['exclude_points'],
        user_data['include_area']
    )
    print(len(output))
    for i in range(len(output)):
        output[i] = pil2np(base642pil(output[i]))

    user_data['output_img'] = output
    return download_img(None, user_data)


with gr.Blocks() as demo:
    user_data = {"origin_img": None,
                 "input_img": None,
                 "include_points": None,
                 "exclude_points": None,
                 "include_area": None,
                 "tmp_img": None,
                 "output_img": None,
                 "size": None
                 }

    stats = gr.State(user_data)

    with gr.Row():
        with gr.Column():
            with gr.Box():
                with gr.Row():
                    input_img = gr.Image(label='输入图像')
                with gr.Row():
                    with gr.Column():
                        include = gr.Textbox(label="包含的点")
                    with gr.Column():
                        exclude = gr.Textbox(label="不包含的点")
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
                    with gr.Column():
                        output_img = gr.Image(label='输出图像1')
                with gr.Row():
                    dw_size = gr.Dropdown(
                        [str(500), str(800), str(1000), str(1500), str(2000)],
                        label="输出尺寸 (n x n), 默认500"
                    )
    with gr.Row():
        with gr.Column():
            with gr.Box():
                with gr.Row():
                    with gr.Column():
                        output_img2 = gr.Image(label='输出图像1')
                    with gr.Column():
                        output_img3 = gr.Image(label='输出图像2')
                    with gr.Column():
                        output_img4 = gr.Image(label='输出图像3')

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
        include
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
        [stats],
        [output_img, output_img2, output_img3, output_img4]
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
