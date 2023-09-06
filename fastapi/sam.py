import uvicorn
from fastapi import FastAPI

from pydantic import BaseModel

from utils import remove_background_img

app = FastAPI()


class FileAccept(BaseModel):
    img: str
    size: int


@app.post("/upload")
async def remove_background(file: FileAccept):
    imgs, scores = remove_background_img(file.size, file.img)

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
    uvicorn.run(app, host="127.0.0.1", port=8001)
