from io import BytesIO
import requests
from matplotlib import pyplot as plt

from image_type import *
from sam_api.config import sam_model_name, sam_model_path
from segment_anything import sam_model_registry, SamPredictor


def get_img_data(img):
    if str(type(img)) == "<class 'numpy.ndarray'>":
        return img
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
    return pil2np(image)


def crop_non_white_region(image):
    """
    将图片裁剪出来
    @param image: PIL格式
    @return: ndarray格式裁剪后图片
    """
    width, height = image.size
    # 获取图像数据
    pixels = image.load()
    # 计算非白色像素的边界框
    left, top, right, bottom = width, height, 0, 0

    for x in range(width):
        for y in range(height):
            # 判断像素是否为白色
            if pixels[x, y] != (255, 255, 255):
                # 更新边界框
                left = min(left, x)
                top = min(top, y)
                right = max(right, x)
                bottom = max(bottom, y)

    # 分割出非白色区域
    delta = 0
    cropped_image = image.crop((left - delta, top - delta, right + delta, bottom + delta))

    return pil2np(cropped_image)


def download_img(imgs, size):
    imgs_base64 = []
    for i in range(len(imgs)):
        imgs[i] = crop_non_white_region(base642pil(imgs[i]))
        h, w = imgs[i].shape[:2]
        scale = min(size / h, size / w)
        resh, resw = int(scale * h), int(scale * w)
        imgs[i] = cv2.resize(imgs[i], (resw, resh))
        imgs_base64.append(pil2base64(np2pil(imgs[i])))

    return imgs_base64


def remove_background_img_sam(size, img, include_point, exclude_point, include_area):
    """
    完成抠图
    @param size: 图像尺寸
    @param img: 输入图像
    @param include_point: 包含的点
    @param exclude_point: 不包含的点
    @param include_area: 矩形框定范围
    @return: 抠图后的图像
    """
    sam = sam_model_registry[sam_model_name](checkpoint=sam_model_path)
    sam.to(device="cuda")
    predictor = SamPredictor(sam)
    image = get_img_data(img)
    # image = cv2.cvtColor(pil2np(content), cv2.COLOR_BGR2RGB)
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

    # 添加总图
    sum_mask = np.logical_or.reduce(masks, axis=0)
    masks = np.concatenate((masks, sum_mask[np.newaxis, :, :]), axis=0)
    scores = np.append(scores, 0)

    imgs = []
    for i, (mask, score) in enumerate(zip(masks, scores)):
        original_pixels = image * np.expand_dims(mask, axis=-1)

        white_pixels = np.ones_like(image) * 255
        white_pixels *= np.expand_dims(np.logical_not(mask), axis=-1)
        result = original_pixels + white_pixels

        removed_img = Image.fromarray(result)
        # 转base64
        pic_base64 = pil2base64(removed_img)
        imgs.append(pic_base64)

    return imgs, scores
