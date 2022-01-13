import cv2
import torch
import os, glob, shutil
import logging
import logging.handlers
from face_ssd_infer import SSD
import tensorflow as tf
from utils import vis_detections, get_face_box_img, get_face_landmark_area_img

logging.FileHandler(filename='Face_detect_crop.log', mode='a', encoding='utf-8')
logging.basicConfig(filename='Face_detect_crop.log', filemode='a', level=logging.INFO)

device = torch.device("cuda")
conf_thresh = 0.3

net = SSD("test")
net.load_state_dict(torch.load('weights/WIDERFace_DSFD_RES152.pth'))
net.to(device).eval()


def detect_output_face_img(input_img_root_path, output_img_root_path):

    if os.path.exists(output_img_root_path) is False:
        os.mkdir(output_img_root_path)

    for parent, dirnames, filenames in os.walk(input_img_root_path):
        for filename in filenames:
            img_path = os.path.join(parent, filename)
            img_out_path = os.path.join(output_img_root_path, filename)
            try:
                img = cv2.imread(img_path)
                curr_image_size = img.shape[0:2]
                target_size = (min(curr_image_size[0], 600), curr_image_size[1] * (min(curr_image_size[0], 600)/curr_image_size[0]))
                detections = net.detect_on_image(img, target_size, device, is_pad=False, keep_thresh=conf_thresh)
                # vis_detections(img, detections, conf_thresh, show_text=False)
                get_face_box_img(img, img_out_path, detections, conf_thresh, show_text=False)
                # logging.info("图片" + img_path + "已截取人脸图")
            except Exception:
                print("图片 " + img_path + " 处理出错")
                logging.info("图片 " + img_path + " 处理出错")

def detect_output_face_landmarkArea_img(input_img_root_path, output_img_root_path, landmarkNet, input_details, output_details):

    if os.path.exists(output_img_root_path) is False:
        os.mkdir(output_img_root_path)

    for parent, dirnames, filenames in os.walk(input_img_root_path):
        for filename in filenames:
            img_path = os.path.join(parent, filename)
            if filename.__contains__(","):
                shutil.move(img_path, (os.path.join(parent, filename.replace(",", "_"))))
                filename = filename.replace(",", "_")
                img_path = os.path.join(parent, filename)

            img_out_path = os.path.join(output_img_root_path, filename)
            try:
                img = cv2.imread(img_path)
                curr_image_size = img.shape[0:2]
                target_size = (min(curr_image_size[0], 800), curr_image_size[1] * (min(curr_image_size[0], 800)/curr_image_size[0]))
                # target_size = (curr_image_size[0], curr_image_size[1])
                detections = net.detect_on_image(img, target_size, device, is_pad=False, keep_thresh=conf_thresh)
                # vis_detections(img, detections, conf_thresh, show_text=False)
                get_face_landmark_area_img(img, img_out_path, detections, landmarkNet, input_details, output_details, conf_thresh, show_text=False)
                # logging.info("图片" + img_path + "已截取人脸图")
            except Exception:
                print("图片 " + img_path + " 处理出错")
                logging.info("图片 " + img_path + " 处理出错")


if __name__ == "__main__":
    input_img_root_path = "input_img_path"
    output_img_root_path = "output_img_path"
    max_resize_value = 600 # GTX1060 6GB RAM
    # max_resize_value = 800 # GTX1660Ti 6GM RAM

    # 人脸区域提取
    detect_output_face_img(input_img_root_path, output_img_root_path)

    # 人脸关键点区域提取
    # load TFLite model and allocate tensors
    lite_filename = 'deploy/landmarks68/pfld_landmarks.tflite'
    interpreter = tf.lite.Interpreter(model_path=lite_filename)
    interpreter.allocate_tensors()
    # get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    detect_output_face_landmarkArea_img(input_img_root_path, output_img_root_path, interpreter, input_details, output_details)
