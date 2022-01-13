# FaceDetection_DSFD_Crop
复杂场景下人脸位置检测与提取

由于所需模型文件较大，需要网盘下载：
WIDERFace_DSFD_RES152.pth链接：https://pan.baidu.com/s/11da8dPjfi930iEfLVvDrmQ 提取码：acu8 
resnet152-b121ed2d.pth链接：链接：https://pan.baidu.com/s/1D6ZrxyyiWbOftJ8HGkzfwg 提取码：5yvb

使用方法：

1. 安装工程运行所需python库
2. 下载主模型文件：WIDERFace_DSFD_RES152.pth（在工程目录下创建weights文件夹，将其拷贝到该目录下）
3. 下载辅助模型文件：resnet152-b121ed2d.pth（第一次运行Demo时，系统会自动创建/home/username/.cache/torch/checkpoints/目录并将模型文件下载到里面，未节省时间可直接将其拷贝到里面）
4. 拷贝需要提取人脸的图片文件夹到指定路径
5. 修改get_face_crop.py代码：根据需要设置以下四个参数：
##### input_img_root_path：待提取人脸原图根目录路径
##### output_img_root_path：提取人脸区域图像目标存储路径
##### max_resize_value：人脸原图缩放边长阈值（建议根据GPU性能设置，如GTX1060 6GB RAM设置为600，GTX1660Ti 6GM RAM设置为800）
##### faceSize_threshold：人脸区域提取边长阈值（过滤掉阈值以下的过小人脸）
6. 运行get_face_crop.py代码（建议在命令行下执行）
