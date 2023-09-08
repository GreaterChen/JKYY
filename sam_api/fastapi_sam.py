import ast

import uvicorn
from fastapi import FastAPI

from pydantic import BaseModel

from utils import remove_background_img_sam

app = FastAPI()


class FileAccept(BaseModel):
    img: str  # base64 or url
    size: int  # size of output image
    include_points: list  # 图像分割部分包含的点
    exclude_points: list  # 图像分割部分不包含的点
    include_area: list  # 图像分割部分包含区域


@app.post("/upload")
async def remove_background(file: FileAccept):
    imgs, scores = remove_background_img_sam(file.size, file.img, file.include_points, file.exclude_points,
                                         file.include_area)
    print(scores)
    res = {
        "main_fig": imgs[-1],
        "sub_figs": [
            {
                "img": img,
                "score": score
            }
            for img, score in zip(imgs[:-1], scores[:-1])
        ]
    }

    return res


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
