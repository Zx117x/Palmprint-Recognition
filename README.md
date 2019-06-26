# Palmprint-Recognition
环境：win10， pytorch 0.41， python 3.6

数据：data下要新建两个文件夹，train和test。
train文件夹下面有256个子文件夹，每个子文件夹下面有5000张照片。
test文件夹下面有256个子文件夹，每个子文件夹下面有1000张照片。
可以运行move_file.py，将train下面的照片随机移动1000张到test下。
使用预训练模型.pth文件放在C盘\.torch\models文件夹下
