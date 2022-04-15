程序改自 https://github.com/enpeizhao/CVprojects

使用opencv自带的SSD网络进行人脸检测，hog特征+SVM进行口罩佩戴情况的分类

## 一、硬件：

* PC端运行：Windows10或11（无需GPU，有最好）或MacOS 都测试可行
* USB RGB 摄像头

## 二、软件：

* Python==3.7
* opencv 

## 三、用法：

### 3.1、电脑运行

* [下载模型](https://github.com/enpeizhao/CVprojects/releases/tag/Models)`face_mask_model.h5`，放到`data目录下`；
* [下载模型](https://github.com/enpeizhao/CVprojects/releases/tag/Models)`res10_300x300_ssd_iter_140000.caffemodel`，放到`weights`目录下；
* 运行下列代码

```
python demosvm.py
```


### 3.3、数据准备及模型训练

* 可以参照 images_preprocess.ipynb





