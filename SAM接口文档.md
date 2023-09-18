# Segment Anything Model 项目接口

### 项目结构

segment_anything

└─sam_api

​	└─config.py	设置SAM模型参数

​	└─fastapi_sam.py	基于fastapi的SAM接口

​	└─gradio_sam.py	基于gradio的SAM图形化界面

​	└─image_type.py	各种图片格式转换函数

​	└─utils.py		若干工具函数

└─checkpoint		存放模型checkpoint，需要在官方github页面中下载

### fastapi接口		

1. 方法：POST

2. url：/upload

3. 请求参数

    |      字段      |         说明         |  类型  |                            备注                             | 是否必填 |
    | :------------: | :------------------: | :----: | :---------------------------------------------------------: | :------: |
    |      img       |      输入的图像      | string |                     支持base64和url格式                     |    是    |
    |      size      |     输出图片尺寸     |  int   |                                                             |    是    |
    | include_points |  目标图中应包含的点  |  list  |                    [[x1,y1],[x2,y2]]格式                    |    否    |
    | exclude_points |  目标图中应排除的点  |  list  |                                                             |    否    |
    |  include_area  | 目标图中应包含的区域 |  list  | [x1,y1,x2,y2]格式；<br />若为空，则上述两个字段至少提供一个 |    否    |

4. 请求示例

```json
{
    "size":500,
    "img":"https://img.evlook.com/evlook/ueeditor/autocoty/131829222851861715.jpg",
    "include_points":[[500, 450]],
    "exclude_points":[],
    "include_area": [200,130,1050,700]
}
```

5. 返回参数

    接口返回三个子图和一个总体图的base64格式信息

```json
"code": 1,
"message": "success",
"data":
{
    "main_fig": img,	// 三个sub_figs的mask堆叠的结果
    "sub_figs": [
        {
            "img": img,		//输出图片的base64
            "score": score	// 置信度
        },
        {
            "img": img,	
            "score": score
        },
        {
            "img": img,	
            "score": score
        }
	]
}
```
### gradio图形界面

![image-20230918153129999](imgs/image-20230918153129999.png)

左上角点击上传图片,上传后点击图片可以选择应包含的点(以红色标识),点击"选择不包含的点"按钮之后,再次点击可以选择应排除的点(以蓝色标识).选择完毕后点击"上传"即可等待结果输出.

![image-20230918153129999](imgs/Snipaste_2023-09-18_15-53-00.png)

输出结果中,页面下面三张图片为SAM模型输出的三种可能的分割结果(在分割小目标时很有用),页面右侧图片为下面三张图片堆叠的结果.选择输出尺寸可以调节图片的大小
