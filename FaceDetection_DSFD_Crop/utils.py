import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import logging
import logging.handlers

logging.FileHandler(filename='Face_detect.log', mode='a', encoding='utf-8')
logging.basicConfig(filename='Face_detect.log', filemode='a', level=logging.INFO)

def vis_detections(im, dets, thresh=0.5, show_text=True):
    """Draw detected bounding boxes."""
    class_name = 'face'
    inds = np.where(dets[:, -1] >= thresh)[0] if dets is not None else []
    if len(inds) == 0:
        return
    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]
        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=2.5)
        )
        if show_text:
            ax.text(bbox[0], bbox[1] - 5,
                    '{:s} {:.3f}'.format(class_name, score),
                    bbox=dict(facecolor='blue', alpha=0.5),
                    fontsize=10, color='white')
    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                 fontsize=10)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('out.png')
    plt.show()

# wyu add
def get_face_box_img(im, outimg, dets, thresh=0.5, show_text=False):
    # class_name = 'face'
    inds = np.where(dets[:, -1] >= thresh)[0] if dets is not None else []
    if len(inds) == 0:
        print("图片 " + outimg + " 中未识别到有效人脸")
        logging.info("图片 " + outimg + " 中未识别到有效人脸")
        return
    # img_size = im.shape
    im = im[:, :, (2, 1, 0)]

    # bbox_ids = []
    # bbox_ws = []

    # fit_face_size_num = 0
    for i in inds:
        bbox = dets[i, :4]
        # score = dets[i, -1]
        w_h = max((bbox[2] - bbox[0]), (bbox[3] - bbox[1]))

        # 多人脸检测
        # 获取人脸框中心点坐标
        face_center = []
        face_center.append((bbox[2] + bbox[0]) / 2)
        face_center.append((bbox[3] + bbox[1]) / 2)
        # # 重新确定人脸框坐标
        new_bbox = []
        new_bbox.append(face_center[0] - (w_h / 2))
        new_bbox.append(face_center[1] - (w_h / 2))
        new_bbox.append(face_center[0] + (w_h / 2))
        new_bbox.append(face_center[1] + (w_h / 2))
        # # 筛选人脸框大小范围
        # if w_h >= 96:
        #     fit_face_size_num=fit_face_size_num+1
        x1 = int(new_bbox[0])# - w_h * 0.05)
        y1 = int(new_bbox[1] + w_h * 0.1)
        x2 = int(new_bbox[2])# + w_h * 0.05)
        y2 = int(new_bbox[3] + w_h * 0.1)

        try:
            region = im[y1:y2, x1:x2]
            if (y2-y1) >= 64 and (x2-x1) >= 64:
                input = cv2.resize(region, (96, 96))
                input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
                if len(inds) == 1:
                    # 单人脸
                    cv2.imwrite(outimg, input)
                    logging.info("图片" + outimg + "已截取单人脸图")
                else:
                    # 对同一张图片多个人脸进行标号存储
                    new_outimg = os.path.splitext(outimg)[0] + "-" + str(i) + os.path.splitext(outimg)[1]
                    cv2.imwrite(new_outimg, input)
                    logging.info("图片" + outimg + "已截取多人脸图：" + new_outimg)
                print("图片 " + outimg + " 中可以输出有效人脸")
            else:
                print("图片 " + outimg + " 中未找到有效人脸：w=" + str(x2-x1) + " h=" + str(y2-y1))
                logging.info("图片 " + outimg + " 中未找到有效人脸：w=" + str(x2-x1) + " h=" + str(y2-y1))
        except Exception as e:
            print("图片 " + outimg + " 获取人脸出错:" + str(e))
            logging.info("图片 " + outimg + " 获取人脸出错:" + str(e))

        # 单人脸检测
    #     bbox_ids.append((i, w_h))
    #     bbox_ws.append(w_h)
    #
    # max_detect_w = max(bbox_ws)
    # for bbox_id in bbox_ids:
    #
    #     if bbox_id[1] == max_detect_w:
    #         bbox = dets[bbox_id[0], :4]
    #
    #         # 获取人脸框中心点坐标
    #         face_center = []
    #         face_center.append((bbox[2] + bbox[0]) / 2)
    #         face_center.append((bbox[3] + bbox[1]) / 2)
    #         # 重新确定人脸框坐标
    #         new_bbox = []
    #         new_x1 = face_center[0] - (max_detect_w / 2)
    #         new_y1 = face_center[1] - (max_detect_w / 2)
    #         new_x2 = face_center[0] + (max_detect_w / 2)
    #         new_y2 = face_center[1] + (max_detect_w / 2)
    #         # max_detect_w = max((new_x2-new_x1),(new_y2-new_y1))
    #         # 边界判断
    #         # if new_x1 < 0 or new_x2 > img_size[0]:
    #         #     if new_x1 < 0 and new_x2 <= img_size[0]:
    #         #         new_x2 = new_x2 + new_x1
    #         #         new_x1 = 0
    #         #     if new_x1 >= 0 and new_x2 > img_size[0]:
    #         #         new_x1 = new_x1 + (new_x2 - img_size[0])
    #         #         new_x2 = img_size[0]
    #         #
    #         #     if new_x1 < 0 and new_x2 > img_size[0]:
    #         #         max_padding = max(abs(new_x1), (new_x2 - img_size[0]))
    #         #         new_x1 = new_x1 + max_padding
    #         #         new_x2 = new_x2 - max_padding
    #         #
    #         #     # max_detect_w = new_x2 - new_x1
    #         #
    #         # if (new_y2 + max_detect_w * 0.1) > img_size[1]:
    #         #     new_y1 = new_y1 + (img_size[1] - new_y2)
    #         #     new_y2 = img_size[1]
    #         # else:
    #         #     new_y1 = new_y1 + max_detect_w * 0.1
    #         #     new_y2 = new_y2 + max_detect_w * 0.1
    #         #
    #         new_bbox.append(new_x1)
    #         new_bbox.append(new_y1)
    #         new_bbox.append(new_x2)
    #         new_bbox.append(new_y2)
    #         #
    #         # x1 = int(new_bbox[0])
    #         # y1 = int(new_bbox[1])
    #         # x2 = int(new_bbox[2])
    #         # y2 = int(new_bbox[3])
    #         x1 = int(new_bbox[0])  # - w_h * 0.05)
    #         y1 = int(new_bbox[1] + max_detect_w * 0.1)
    #         x2 = int(new_bbox[2])# + w_h * 0.05)
    #         y2 = int(new_bbox[3] + max_detect_w * 0.1)
    #
    #         try:
    #             region = im[y1:y2, x1:x2]
    #             if (y2-y1) >= 96 and (x2-x1) >= 96:
    #                 # input = cv2.resize(region, (96, 96))
    #                 input = cv2.cvtColor(region, cv2.COLOR_BGR2RGB)
    #                 # 对同一张图片多个人脸进行标号存储
    #                 # new_outimg = os.path.splitext(outimg)[0] + "_" + str(i) + os.path.splitext(outimg)[1]
    #                 cv2.imwrite(outimg, input)
    #                 print("图片 " + outimg + " 中可以输出有效人脸")
    #             else:
    #                 print("图片 " + outimg + " 中未找到有效人脸：w=" + str(x2-x1) + " h=" + str(y2-y1))
    #         except Exception:
    #             print("图片 " + outimg + " 获取人脸出错")

    # print("图片 " + outimg + " 中检测到有效人脸数为 " + str(fit_face_size_num))

def get_face_landmark_area_img(im, outimg, dets, landmarkNet, input_details, output_details, thresh=0.5, show_text=False):
    # class_name = 'face'
    inds = np.where(dets[:, -1] >= thresh)[0] if dets is not None else []
    if len(inds) == 0:
        print("图片 " + outimg + " 中未识别到有效人脸")
        logging.info("图片 " + outimg + " 中未识别到有效人脸")
        return
    # img_size = im.shape
    im = im[:, :, (2, 1, 0)]

    # bbox_ids = []
    # bbox_ws = []

    # fit_face_size_num = 0
    for i in inds:
        bbox = dets[i, :4]
        # score = dets[i, -1]
        w_h = max((bbox[2] - bbox[0]), (bbox[3] - bbox[1]))

        # 多人脸检测
        # 获取人脸框中心点坐标
        face_center = []
        face_center.append((bbox[2] + bbox[0]) / 2)
        face_center.append((bbox[3] + bbox[1]) / 2)
        # # 重新确定人脸框坐标
        new_bbox = []
        new_bbox.append(face_center[0] - (w_h / 2))
        new_bbox.append(face_center[1] - (w_h / 2))
        new_bbox.append(face_center[0] + (w_h / 2))
        new_bbox.append(face_center[1] + (w_h / 2))
        # # 筛选人脸框大小范围
        # if w_h >= 96:
        #     fit_face_size_num=fit_face_size_num+1
        x1 = int(new_bbox[0])# - w_h * 0.05)
        y1 = int(new_bbox[1] + w_h * 0.1)
        x2 = int(new_bbox[2])# + w_h * 0.05)
        y2 = int(new_bbox[3] + w_h * 0.1)

        try:
            region = im[y1:y2, x1:x2]
            if (y2-y1) >= 64 and (x2-x1) >= 64:
                # input = cv2.resize(region, (112, 112))
                input = cv2.cvtColor(region, cv2.COLOR_BGR2RGB)

                faceLandmark = get68Landmarks(input, landmarkNet, input_details, output_details)

                faceLandmarkArea = get68LandmarksArea(im, faceLandmark, x1, y1)

                faceLandmarkArea = cv2.cvtColor(faceLandmarkArea, cv2.COLOR_BGR2RGB)
                faceLandmarkArea = cv2.resize(faceLandmarkArea, (96, 96))

                if len(inds) == 1:
                    # 单人脸
                    cv2.imwrite(outimg, faceLandmarkArea)
                    logging.info("图片" + outimg + "已截取单人脸图")
                else:
                    # 对同一张图片多个人脸进行标号存储
                    new_outimg = os.path.splitext(outimg)[0] + "-" + str(i) + os.path.splitext(outimg)[1]
                    cv2.imwrite(new_outimg, faceLandmarkArea)
                    logging.info("图片" + outimg + "已截取多人脸图：" + new_outimg)
                print("图片 " + outimg + " 中可以输出有效人脸")
            else:
                print("图片 " + outimg + " 中未找到有效人脸：w=" + str(x2-x1) + " h=" + str(y2-y1))
                logging.info("图片 " + outimg + " 中未找到有效人脸：w=" + str(x2-x1) + " h=" + str(y2-y1))
        except Exception as e:
            print("图片 " + outimg + " 获取人脸出错:" + str(e))
            logging.info("图片 " + outimg + " 获取人脸出错:" + str(e))

def bbox_vote(det):
    order = det[:, 4].ravel().argsort()[::-1]
    det = det[order, :]
    dets = None
    while det.shape[0] > 0:
        # IOU
        area = (det[:, 2] - det[:, 0] + 1) * (det[:, 3] - det[:, 1] + 1)
        xx1 = np.maximum(det[0, 0], det[:, 0])
        yy1 = np.maximum(det[0, 1], det[:, 1])
        xx2 = np.minimum(det[0, 2], det[:, 2])
        yy2 = np.minimum(det[0, 3], det[:, 3])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        o = inter / (area[0] + area[:] - inter)
        # get needed merge det and delete these det
        merge_index = np.where(o >= 0.3)[0]
        det_accu = det[merge_index, :]
        det = np.delete(det, merge_index, 0)
        if merge_index.shape[0] <= 1:
            continue
        det_accu[:, 0:4] = det_accu[:, 0:4] * np.tile(det_accu[:, -1:], (1, 4))
        max_score = np.max(det_accu[:, 4])
        det_accu_sum = np.zeros((1, 5))
        det_accu_sum[:, 0:4] = np.sum(det_accu[:, 0:4], axis=0) / np.sum(det_accu[:, -1:])
        det_accu_sum[:, 4] = max_score
        try:
            dets = np.row_stack((dets, det_accu_sum))
        except:
            dets = det_accu_sum
    if dets is not None:
        dets = dets[0:750, :]
    return dets


def add_borders(curr_img, target_shape=(224, 224), fill_type=0):
    curr_h, curr_w = curr_img.shape[0:2]
    shift_h = max(target_shape[0] - curr_h, 0)
    shift_w = max(target_shape[1] - curr_w, 0)

    image = cv2.copyMakeBorder(curr_img, shift_h // 2, (shift_h + 1) // 2, shift_w // 2, (shift_w + 1) // 2, fill_type)
    return image, shift_h, shift_w


def resize_image(image, target_size, resize_factor=None, is_pad=True, interpolation=3):
    curr_image_size = image.shape[0:2]

    if resize_factor is None and is_pad:
        resize_factor = min(target_size[0] / curr_image_size[0], target_size[1] / curr_image_size[1])
    elif resize_factor is None and not is_pad:
        resize_factor = np.sqrt((target_size[0] * target_size[1]) / (curr_image_size[0] * curr_image_size[1]))

    image = cv2.resize(image, None, None, fx=resize_factor, fy=resize_factor, interpolation=interpolation)

    if is_pad:
        image, shift_h, shift_w = add_borders(image, target_size)
    else:
        shift_h = shift_w = 0

    scale = np.array([image.shape[1]/resize_factor, image.shape[0]/resize_factor,
                      image.shape[1]/resize_factor, image.shape[0]/resize_factor])

    return image, shift_h/image.shape[0]/2, shift_w/image.shape[1]/2, scale

def get68Landmarks(faceCrop, pfldNet, input_details, output_details):

    # input = cv2.cvtColor(faceCrop, cv2.COLOR_BGR2RGB)
    input = cv2.resize(faceCrop, (112, 112))
    input = input.astype(np.float32) / 256.0
    input = np.expand_dims(input, 0)

    pfldNet.set_tensor(input_details[0]['index'], input)
    pfldNet.invoke()

    pre_landmarks = pfldNet.get_tensor(output_details[0]['index'])
    pre_landmark = pre_landmarks[0]

    height, width, _ = faceCrop.shape
    pre_landmark = pre_landmark.reshape(-1, 2) * [height, width]
    return pre_landmark

def get68LandmarksArea(img, faceLandmark, x1, y1):
    xy = np.min(faceLandmark, axis=0).astype(np.int32)
    zz = np.max(faceLandmark, axis=0).astype(np.int32)
    landmark_x1 = x1 + xy[0]
    # landmark_x1 = x1 + xy[0]
    # landmark_x2 = x1 + zz[0]
    # landmark_x2 = x1 + zz[0]
    landmark_y1 = y1 + xy[1]
    # landmark_y1 = y1 + xy[1]
    # landmark_y2 = y1 + zz[1]
    # landmark_y2 = y1 + zz[1]

    old_image_copy = img.copy()

    # make landmark region is square
    height2, width2 = (zz[1] - xy[1] + 1), (zz[0] - xy[0] + 1)
    region_size = int(max([width2, height2]) * 1.0)
    region_cx = landmark_x1 + width2 // 2
    region_cy = landmark_y1 + height2 // 2
    region_x1 = region_cx - region_size // 2
    region_x2 = region_x1 + region_size
    region_y1 = region_cy - region_size // 2
    region_y2 = region_y1 + region_size

    region_x1 = max(0, region_x1)
    region_y1 = max(0, region_y1)

    landmark_region = old_image_copy[region_y1:region_y2, region_x1:region_x2]

    return landmark_region