from io import BytesIO
import requests
from matplotlib import pyplot as plt

from image_type import *
from segment_anything import sam_model_registry, SamPredictor


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


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)


def remove_background_img(size, img, point):
    sam = sam_model_registry["vit_h"](checkpoint="../checkpoint/sam_vit_h_4b8939.pth")
    sam.to(device="cuda")
    predictor = SamPredictor(sam)
    content = get_img_data(img)
    image = cv2.cvtColor(pil2np(content), cv2.COLOR_BGR2RGB)

    predictor.set_image(image)
    input_point = []
    input_label = []
    for x, y in point:
        input_point.append([x, y])
        input_label.append(1)

    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    show_points(np.array(input_point), np.array(input_label), plt.gca())
    plt.axis('off')
    plt.show()

    masks, scores, logits = predictor.predict(
        point_coords=np.array(input_point),
        point_labels=np.array(input_label),
        multimask_output=True,
    )

    sum_mask = np.logical_or.reduce(masks, axis=0)
    masks = np.concatenate((masks, sum_mask[np.newaxis, :, :]), axis=0)
    scores = np.append(scores, 0)

    sub_imgs = []
    for i, (mask, score) in enumerate(zip(masks, scores)):
        original_pixels = image * np.expand_dims(mask, axis=-1)

        white_pixels = np.ones_like(image) * 255
        white_pixels *= np.expand_dims(np.logical_not(mask), axis=-1)

        result = original_pixels + white_pixels

        removed_img = Image.fromarray(result)
        h, w = image.shape[:2]
        scale = min(size / h, size / w)
        resh, resw = int(scale * h), int(scale * w)
        res_img = removed_img.resize((resw, resh))
        res_img.show()

        pic_base64 = pil2base64(res_img)
        sub_imgs.append(pic_base64)

    return sub_imgs, scores
