import os
import math
import cv2
import numpy as np
import gradio as gr
import rembg
import warnings

from image_type import np2pil, pil2np, base642pil
from sam_api.utils import remove_background_img_sam, crop_non_white_region

warnings.filterwarnings("ignore")
IMG_MAX_SIZE = 500


def get_preview(img):
    # 在此裁剪会导致选点不准，如果想裁剪需要对点再进行同步偏移
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

    return get_preview(user_data['input_img']), [], [], 0


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
    user_data['include_points'] = []
    user_data['exclude_points'] = []

    return get_preview(user_data['input_img']), 0, [], []


def flip_img(tmp_img_angle, user_data):
    '''
    Flip image.
    '''
    # Check if input image is None
    if user_data['input_img'] is None:
        # Return None if input image is None
        return None

    # Copy input image
    img = user_data['input_img'].copy()

    # Flip image
    img = cv2.flip(img, 1)
    # Update temporary image
    user_data['tmp_img'] = img

    # Get height and width of image
    h, w = img.shape[:2]
    # Get center of rotation
    rotate_center = (w / 2, h / 2)
    # Get rotation matrix
    M = cv2.getRotationMatrix2D(rotate_center, tmp_img_angle, 1.0)
    # Get new width
    new_w = int(h * np.abs(M[0, 1]) + w * np.abs(M[0, 0]))
    # Get new height
    new_h = int(h * np.abs(M[0, 0]) + w * np.abs(M[0, 1]))
    # Set rotation matrix to 0, 0, 1
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2
    # Apply rotation matrix to image
    user_data['tmp_img'] = cv2.warpAffine(img, M, (new_w, new_h))

    # Return preview of new image
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


def mode_change(check_box, mode):
    mode = check_box
    return mode


def get_point(evt: gr.SelectData, user_data, mode):
    if user_data['input_img'] is None:
        return None

    img = user_data['input_img'].copy()
    if mode:
        user_data['exclude_points'].append(evt.index)
        cv2.circle(img, evt.index, img.shape[0]//100, (0, 0, 255), -1)
    else:
        user_data['include_points'].append(evt.index)
        cv2.circle(img, evt.index, img.shape[0]//100, (255, 0, 0), -1)

    user_data['input_img'] = img

    return user_data['input_img'], user_data['include_points'], user_data['exclude_points']


def download_img(size, user_data):
    try:
        size = int(size)
    except:
        size = 500

    if user_data['output_img'] is None:
        return None

    imgs = user_data['output_img']
    for i in range(len(imgs)):
        imgs[i] = crop_non_white_region(np2pil(imgs[i]))
        h, w = imgs[i].shape[:2]
        scale = min(size / h, size / w)
        resh, resw = int(scale * h), int(scale * w)
        imgs[i] = cv2.resize(imgs[i], (resw, resh))

    return imgs[-1], imgs[0], imgs[1], imgs[2]


def remove_background_img(user_data):
    if user_data['tmp_img'] is None:
        return None
    output, scores = remove_background_img_sam(
        user_data['size'],
        user_data['tmp_img'],
        user_data['include_points'],
        user_data['exclude_points'],
        user_data['include_area']
    )
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

    mode = False
    stats = gr.State(user_data)
    mode = gr.State(mode)

    with gr.Row():
        with gr.Column():
            with gr.Box():
                with gr.Row():
                    input_img = gr.Image(label='输入图像')
                with gr.Row():
                    in_or_ex = gr.Checkbox(label="选择不包含的点")
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
                        output_img = gr.Image(label='输出图像叠加')
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
        [input_img, include, exclude, angle]
    )

    input_img.clear(
        clear,
        [stats],
        input_img
    )

    input_img.select(
        get_point,
        [stats, mode],
        [input_img, include, exclude]
    )

    reset.click(
        reset_img,
        [stats],
        [input_img, angle, include, exclude]
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
        [output_img, output_img2, output_img3, output_img4]
    )

    in_or_ex.change(
        mode_change,
        [in_or_ex, mode],
        mode
    )

if __name__ == '__main__':
    demo.queue().launch(share=False, inbrowser=True,
                        server_name="127.0.0.1",
                        server_port=18401
                        # root_path="/HeadView/Web"
                        )
