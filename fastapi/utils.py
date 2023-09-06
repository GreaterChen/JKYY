from io import BytesIO
import requests

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


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    mask = ~mask
    h, w = mask.shape[-2:]

    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)

    ax.imshow(mask_image)


def remove_background_img(size, img):
    sam = sam_model_registry["vit_h"](checkpoint="../checkpoint/sam_vit_h_4b8939.pth")
    sam.to(device="cuda")
    predictor = SamPredictor(sam)
    content = get_img_data(img)
    image = cv2.cvtColor(pil2np(content), cv2.COLOR_BGR2RGB)

    predictor.set_image(image)

    input_point = np.array([[image.shape[0] // 2, image.shape[0] // 2]])
    input_label = np.array([1])

    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
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
