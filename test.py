import base64

import cv2
import os

import numpy as np
from matplotlib import pyplot as plt

from image_type import pil2np
from sam_api.utils import get_img_data
from segment_anything import sam_model_registry, SamPredictor


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([1, 1, 1, 0])
    mask = ~mask
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


sam = sam_model_registry["vit_b"](checkpoint="checkpoint/sam_vit_b_01ec64.pth")
sam.to(device="cuda")
predictor = SamPredictor(sam)  # SAM预测图像

# img = cv2.imread("input/truck.jpg")
with open('input/truck.jpg', 'rb') as file:
    pic_base64 = str(base64.b64encode(file.read()), encoding='utf-8')
content = get_img_data(pic_base64)
image = cv2.cvtColor(pil2np(content), cv2.COLOR_BGR2RGB)
predictor.set_image(image)

input_point = np.array([[500, 375], [1125, 625]])
input_label = np.array([1, 1])

masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True,
)

for i, (mask, score) in enumerate(zip(masks, scores)):
    plt.figure(figsize=(10,10))
    plt.imshow(image)
    show_mask(mask, plt.gca())
    show_points(input_point, input_label, plt.gca())
    plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
    plt.axis('off')
    plt.show()

for i, (mask, score) in enumerate(zip(masks, scores)):
    mask = ~mask
    mask = mask + 255
    mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    mask = mask.astype(np.uint8)
    res = cv2.bitwise_and(image, mask)
    res[res == 0] = 255
    plt.imshow(res)
    plt.savefig('res-{}.png'.format(i + 1))
    plt.show()

