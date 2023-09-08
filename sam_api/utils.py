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
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='blue', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


def remove_background_img_sam(size, img, include_point, exclude_point, include_area):
    """
    完成抠图
    @param size: 输出图像尺寸
    @param img: 输出图像base64
    @param include_point: 包含的点
    @param exclude_point: 不包含的点
    @param include_area: 矩形框定范围
    @return: 抠图后的图像
    """
    sam = sam_model_registry["vit_h"](checkpoint="../checkpoint/sam_vit_h_4b8939.pth")
    sam.to(device="cuda")
    predictor = SamPredictor(sam)
    content = get_img_data(img)
    # image = cv2.cvtColor(pil2np(content), cv2.COLOR_BGR2RGB)
    image = pil2np(content)
    predictor.set_image(image)

    input_point = []
    input_label = []
    for x, y in include_point:
        input_point.append([x, y])
        input_label.append(1)

    for x, y in exclude_point:
        input_point.append([x, y])
        input_label.append(0)

    input_point = np.array(input_point)
    input_label = np.array(input_label)
    include_area = np.array(include_area)

    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        # box=include_area[None, :],
        multimask_output=True,
    )

    # 测试时展示选定点和区域
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    # show_box(include_area, plt.gca())
    show_points(input_point, input_label, plt.gca())
    plt.axis('off')
    plt.show()

    # 添加总图
    sum_mask = np.logical_or.reduce(masks, axis=0)
    masks = np.concatenate((masks, sum_mask[np.newaxis, :, :]), axis=0)
    scores = np.append(scores, 0)

    imgs = []
    for i, (mask, score) in enumerate(zip(masks, scores)):
        if i != 3:
            continue
        original_pixels = image * np.expand_dims(mask, axis=-1)

        white_pixels = np.ones_like(image) * 255
        white_pixels *= np.expand_dims(np.logical_not(mask), axis=-1)
        result = original_pixels + white_pixels

        # 进行裁剪
        removed_img = Image.fromarray(result)
        h, w = image.shape[:2]
        scale = min(size / h, size / w)
        resh, resw = int(scale * h), int(scale * w)
        res_img = removed_img.resize((resw, resh))
        # res_img.show()

        # 转base64
        pic_base64 = pil2base64(res_img)
        imgs.append(pic_base64)

    return imgs, scores
