from __future__ import print_function
import os
import shutil
import glob
import filecmp
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import cv2
import time
from face_detection.face_detection import Face
from PIL import Image
import logging
import logging.handlers

logging.FileHandler(filename='data_clean.log', mode='a', encoding='utf-8')
logging.basicConfig(filename='data_clean.log', filemode='a', level=logging.INFO)

# get face region from image and save with file
def getAndSaveFaceImgs(input_img_root_path, output_img_root_path, vis_threshold, faceSize_threshold):
    faceboxes = Face()
    if (os.path.exists(output_img_root_path)) is False:
        os.makedirs(output_img_root_path)
    for parent, dirnames, filenames in os.walk(input_img_root_path):
        for filename in filenames:
            img_path = os.path.join(parent, filename)
            img_out_path = os.path.join(output_img_root_path, filename)
            try:
                img = cv2.imread(img_path)
                im_height, im_width, _ = img.shape

                # start_time = time.time()
                boxes = faceboxes.face_detection(img)

                img_show = img.copy()
                # get and save image
                i = 0
                for b in boxes:
                    x1, y1, x2, y2, score = int(b[0]), int(b[1]), int(b[2]), int(b[3]), b[4]
                    if score < vis_threshold:
                        continue

                    w = x2 - x1 + 1
                    h = y2 - y1 + 1

                    size = int(max([w, h]) * 1.0)
                    cx = x1 + w // 2
                    cy = y1 + h // 2
                    x1 = cx - size // 2
                    x2 = x1 + size
                    y1 = cy - size // 2
                    y2 = y1 + size

                    dx = max(0, -x1)
                    dy = max(0, -y1)
                    x1 = max(0, x1)
                    y1 = max(0, y1)

                    edx = max(0, x2 - im_width)
                    edy = max(0, y2 - im_height)
                    x2 = min(im_width, x2)
                    y2 = min(im_height, y2)

                    try:
                        if (dx > 0 or dy > 0 or edx > 0 or edy > 0):
                            print('copyMakeBorder')
                            print('dy, edy, dx, edx: ', dy, edy, dx, edx)
                            if dx < 0:
                                dx = 0
                            if dy < 0:
                                dy = 0
                            if edx < 0:
                                edx = 0
                            if edy < 0:
                                edy = 0
                            img_show = cv2.copyMakeBorder(img, dy, edy, dx, edx, cv2.BORDER_CONSTANT, value=[0, 0, 0])
                            region = img_show[y1:y2+edx, x1:x2+edy]
                        else:
                            region = img_show[y1:y2, x1:x2]
                        if (y2 - y1) >= faceSize_threshold and (x2 - x1) >= faceSize_threshold:
                            if len(boxes) == 1:
                                # 单人脸
                                cv2.imwrite(img_out_path, region)
                            else:
                                # 对同一张图片多个人脸进行标号存储
                                new_outimg = os.path.splitext(img_out_path)[0] + "-" + str(i) + os.path.splitext(img_out_path)[1]
                                cv2.imwrite(new_outimg, region)
                                i = i+1
                            print("图片 " + img_out_path + " 中可以输出有效人脸")
                        else:
                            print("图片 " + img_out_path + " 中未找到有效人脸：w=" + str(x2 - x1) + " h=" + str(y2 - y1))
                    except Exception as e:
                        print("图片 " + img_out_path + " 获取人脸出错:" + str(e))
            except Exception:
                print("图片 " + img_path + " 处理出错")


if __name__ == '__main__':
    wait_crop_img_root_path = "wait_crop_img_root_path"
    target_crop_img_dir_path = "target_crop_img_dir_path"
    face_vis_threshold = 0.7
    faceSize_threshold = 96

    getAndSaveFaceImgs(wait_crop_img_root_path, target_crop_img_dir_path, face_vis_threshold, faceSize_threshold)


