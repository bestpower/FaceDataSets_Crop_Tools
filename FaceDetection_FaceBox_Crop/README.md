# FaceDetection_FaceBox_Crop
轻量人脸位置检测与提取

使用方法：

1. 安装工程运行所需python库
2. 拷贝需要提取人脸的图片文件夹到指定路径
3. 修改get_face_crop.py代码：根据需要设置以下四个参数：
##### wait_crop_img_root_path：待提取人脸原图根目录路径
##### target_crop_img_dir_path：提取人脸区域图像目标存储路径
##### face_vis_threshold：人脸检测阈值（建议设置为0.7）
##### faceSize_threshold：人脸区域提取边长阈值（过滤掉阈值以下的过小人脸）
4. 运行get_face_crop.py代码（建议在命令行下执行）
