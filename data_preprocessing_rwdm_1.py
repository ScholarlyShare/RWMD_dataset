'''
Descripttion: This code is used for the pre-processing of the RWMD dataset
version: 1.0
Author: duany
Date: 2024-07-29 03:47:03
LastEditors: xuzhen
LastEditTime: 2024-07-29 22:29:50
'''
import os
import cv2
from tqdm import tqdm
import shutil
import random
import numpy as np
import glob
import json
from base64 import b64encode, b64decode
from io import BytesIO
from PIL import Image

def change_max_size(src_dir):
    dir_match = [("IMG", "IMG_RESIZE"), ("GT", "GT_RESIZE")]
    for src_name, dst_name in dir_match:
        src_path = os.path.join(src_dir, src_name)
        dst_path = os.path.join(src_dir, dst_name)
        
        for file_name in tqdm([name for name in os.listdir(src_path) if name.endswith(".png") or name.endswith(".jpg")]):
            src_file = os.path.join(src_path, file_name)
            dst_file = os.path.join(dst_path, file_name)
            
            if "GT" in src_name:
                read_flag = cv2.IMREAD_GRAYSCALE
                resize_flag = cv2.INTER_NEAREST
            else:
                read_flag = cv2.IMREAD_COLOR
                resize_flag = cv2.INTER_AREA
            img = cv2.imread(src_file, read_flag)
            # resize the max size to 1500
            max_size = max(img.shape[0], img.shape[1])
            if max_size > 1500:
                scale = 1500 / max_size
                img = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)), interpolation=resize_flag)
                cv2.imwrite(dst_file, img)

def choise_datas(src_dir, dst_dir, num=100):
    img_dir = os.path.join(src_dir, "img")
    gt_dir = os.path.join(src_dir, "mask")
    img_name = [name for name in os.listdir(img_dir) if not name.endswith("_4.png") and not name.endswith("_1.png") and not name.endswith("_2.png") and not name.endswith("_3.png")]
    random.shuffle(img_name)
    img_name = img_name[:num]
    for name in tqdm(img_name):
        img_path = os.path.join(img_dir, name)
        mask_path = os.path.join(gt_dir, name)
        shutil.copy(img_path, os.path.join(dst_dir, "img"))
        shutil.copy(mask_path, os.path.join(dst_dir, "mask"))

def check_labels(src_dir):
    gt_dir = os.path.join(src_dir, "mask")
    gt_names = [name for name in os.listdir(gt_dir)]
    label_set = set()
    for gt_n in tqdm(gt_names):
        gt_path = os.path.join(gt_dir, gt_n)
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        labels = np.unique(gt)
        label_set.update(labels)
    print(label_set)
    


def cp2all(src_dir_list, dst_dir):
    for src_dir in src_dir_list:
        img_dir = os.path.join(src_dir, "img")
        gt_dir = os.path.join(src_dir, "mask")
        img_name = [name for name in os.listdir(img_dir)]
        # img_name = [name for name in os.listdir(img_dir) if not name.endswith("_4.png") and not name.endswith("_1.png") and not name.endswith("_2.png") and not name.endswith("_3.png")]
        for name in tqdm(img_name):
            img_path = os.path.join(img_dir, name)
            mask_path = os.path.join(gt_dir, name)
            shutil.copy(img_path, os.path.join(dst_dir, "img"))
            shutil.copy(mask_path, os.path.join(dst_dir, "mask"))

def resize(src_dir, dst_dir):
    imgs_paths = glob.glob(src_dir + "/img/*png")
    for img_p in tqdm(imgs_paths):
        img = cv2.imread(img_p)
        img_n = os.path.basename(img_p)
        label_p = os.path.join(src_dir, "mask", img_n)
        mask = cv2.imread(label_p, cv2.IMREAD_GRAYSCALE)
        # 最大边缩放到1024, scale the longest edge of the image to 1024
        max_size = max(img.shape[0], img.shape[1])
        if max_size > 1024:
            scale = 1024 / max_size
            img = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, (int(mask.shape[1] * scale), int(mask.shape[0] * scale)), interpolation=cv2.INTER_NEAREST)
        
        cv2.imwrite(os.path.join(dst_dir, "img", img_n), img)
        cv2.imwrite(os.path.join(dst_dir, "mask", img_n), mask)
        
def savePoints(src_dir):
    savePointsWithResize(src_dir, None)

def savePointsWithResize(src_dir, edge_limit=1024):
    label_points = {}
    imgs_path = glob.glob(src_dir + "/*[jp][pn]g")
    for img_p in tqdm(imgs_path):
        img = cv2.imread(img_p)
        img_n = os.path.basename(img_p)
        label_p = img_p.replace(".jpg", ".json") if "jpg" in img_n else  img_p.replace(".png", ".json")
        label = json.load(open(label_p, "r"), encoding="utf-8")
        
        for l in label['shapes']:
            label_symbol = l['label']      # str
            if "foreground_doc" == label_symbol:
                points = np.asarray(l['points']).astype(np.float32)
                if edge_limit is not None:
                    # 最大边缩放到1024, scale the longest edge of the image to 1024
                    max_size = max(img.shape[0], img.shape[1])
                    if max_size > 1024:
                        scale = 1024 / max_size
                    else:
                        scale = 1.0
                    points = points * scale
                points = points.astype(np.int32)           # (N, 2)
                img_k = os.path.splitext(img_n)[0]
                label_points[img_k] = points.tolist()
                break
    json.dump(label_points, open("label_points.json", "w"), indent=4, ensure_ascii=False)
 
def test_rotate3D():
    from datas_augment.tools.math_utils import PerspectiveTransform, cliped_rand_norm
    x = cliped_rand_norm(0, 180 / 4)
    y = cliped_rand_norm(0, 180 / 4)
    z = cliped_rand_norm(0, 180 / 4)
    trans = PerspectiveTransform(x, y, z, 1.0, 50)
    img = cv2.imread("/data3/duanyong2/datas/competation/FloodNet/val/IMG/10833.jpg")
    h, w, _ = img.shape
    points_src = [[10, 10], [w - 100, 100], [w - 400, h - 400], [100, h - 200]]
    img_src = img.copy()
    # 画点,draw points
    for p in points_src:
        cv2.circle(img_src, p, 50, (0, 0, 255), -1)
    cv2.imwrite("img_src.png", img_src)
    # dst, M33, ptsOut = trans.transform_image(img)
    heatmap = np.zeros((h, w), dtype=np.uint8)
    dst, M33, ptsOut, heatmap = trans.transform_image_with_heatmap(img, heatmap)
    points_dst = trans.transform_pnts([points_src], M33)
    
    left = min([int(points[0]) for points in ptsOut])
    top = min([int(points[1]) for points in ptsOut])
    right = max([int(points[0]) for points in ptsOut])
    bottom = max([int(points[1]) for points in ptsOut])
    result_img = dst[top:bottom, left:right, :]
    
    if points_dst is not None and len(points_dst) > 0:
        points_dst[:, :, 0] -= left
        points_dst[:, :, 1] -= top

    for p in points_dst[0]:
        cv2.circle(result_img, list(map(int, p)), 50, (0, 0, 255), -1)
    cv2.imwrite("img_dst.png", result_img)


def increase_exampaper(src_dir, dst_dir, expand=7):
    name_exampaper = ['edge_4', 'edge_5', 'edge_6', 'edge_7', 'edge_8', 'edge_15', 'edge_16', 'edge_35', 'edge_39', 'edge_455']
    paths_img = glob.glob(src_dir + "/img/*[jp][pn]g")
    path_label = os.path.join(src_dir, "label_points.json")
    label = json.load(open(path_label, "r"), encoding="utf-8")
    for path in tqdm(paths_img):
        name = os.path.split(path)[-1]
        path_img = os.path.join(src_dir, "img", name)
        name_k = os.path.splitext(name)[0]
        path_mask = os.path.join(src_dir, "mask", name_k+".png")
        if name_k in name_exampaper:
            for i in range(expand):
                name_expand_k = name_k + "_" + str(i)
                name_expand = name_expand_k + ".png"
                shutil.copy(path_img, os.path.join(dst_dir, "img", name_expand))
                shutil.copy(path_mask, os.path.join(dst_dir, "mask", name_expand))
                assert name_expand_k not in label, f"{name_expand_k} should not in label keys"
                label[name_expand_k] = label[name_k]
    json.dump(label, open(os.path.join(dst_dir, "label_points.json"), "w"), indent=4, ensure_ascii=False)

def test_shape(mask_path1, mask_path2):
    img1_cv = cv2.imread(mask_path1, cv2.IMREAD_GRAYSCALE)
    img2_cv = cv2.imread(mask_path2, cv2.IMREAD_GRAYSCALE)
    print(img1_cv.shape, img2_cv.shape)
    # for name in os.listdir(mask_dir):
    #     mask_p = os.path.join(mask_dir, name)
    #     img = cv2.imread(mask_p, cv2.IMREAD_GRAYSCALE)
    #     if len(img.shape) == 3:
    #         print(img.shape)
    #     else:
    #         print(img.shape)
    from detectron2.data import detection_utils as utils
    img1 = utils.read_image(mask_path1, "L")
    img1 = np.squeeze(img1, -1)
    img2 = utils.read_image(mask_path2)
    print(img1.shape, img2.shape)


def statistics_label_v2(mask_dir):
    max_num_dict = {}
    for mask_name in tqdm(os.listdir(mask_dir)):
        mask_p = os.path.join(mask_dir, mask_name)
        mask = cv2.imread(mask_p, cv2.IMREAD_GRAYSCALE)
        labels = list(np.unique(mask))
        if 0 in labels:
            labels.remove(0)
        for l in labels:
            if l not in max_num_dict:
                max_num_dict[l] = 1
            else:
                max_num_dict[l] += 1
    print(max_num_dict)


def genarate_label_from_ori(src_dir, dst_dir):
    """json标注的数据格式，转换成mask格式"""
    "To convert the original json label data to the mask image"
    label_mapper = {"1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "10": 10,
                     "11": 11, "12": 12, "13": 13, "14": 14, "15": 15, "16": 16, "17": 17, "18": 18}
    
    points_json = {}
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)
    if not os.path.exists(dst_dir + "/img"):
        os.mkdir(dst_dir+"/img")
    if not os.path.exists(dst_dir + "/mask"):
        os.mkdir(dst_dir+"/mask")


    # for img_p in tqdm(imgs_path):
    for root, dirs, files in os.walk(src_dir):
        for file in tqdm(files):
            # TODO delete
            # if os.path.splitext(file)[0] in ['sanxingS21ultra-1984', 'sanxingS21ultra-1949', 'sanxingS21ultra-1466', 'sanxingS21ultra-1950']:
            #     continue

            if file.endswith(".json"):
                continue
            img_p = os.path.join(root, file)
            img = cv2.imread(img_p)
            img_n = os.path.basename(img_p)
            if "jpg" in img_n:
                label_p = img_p.replace(".jpg", ".json")
            elif "png" in img_n:
                label_p = img_p.replace(".png", ".json")
            elif "jpeg" in img_n:
                label_p = img_p.replace(".jpeg", ".json")
            elif "JPG" in img_n:
                label_p = img_p.replace(".JPG", ".json")
            elif "PNG" in img_n:
                label_p = img_p.replace(".PNG", ".json")
            elif "JPEG" in img_n:
                label_p = img_p.replace(".JPEG", ".json")
            label = json.load(open(label_p, "r"), encoding="utf-8")
            
            label_infos = []
            foreground_quad = []
            for l in label['shapes']:
                label_symbol = l['label']      # str
                if label_symbol not in label_mapper:
                    if label_symbol == "foreground_doc":
                        foreground_quad = np.asarray(l['points']).astype(np.int32)
                    continue
                label_idx = label_mapper[label_symbol]
                points = np.asarray(l['points']).astype(np.int32)           # (N, 2)
                label_infos.append((label_idx, points))

            mask = np.zeros(img.shape[:2], dtype=np.uint8)
            label_infos_sorted = sorted(label_infos, key=lambda x: x[0])
            # 找到前景id, find the foreground document id
            fore_idx = label_infos_sorted[-1][0]
            # 用1临时填充背景mask, temporarily fill the background mask with 1
            for label_idx, points in label_infos_sorted[:-1]:
                cv2.fillPoly(mask, [points], 1)
            # 重新给背景文档赋值, reassign the background document
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            too_small_contours = 0
            new_label_infos = []
            new_label_idx = 0
            for i in range(len(contours)):
                hull = cv2.convexHull(contours[i])
                area_cur = cv2.contourArea(hull)
                if area_cur < 100:
                    too_small_contours += 1
                    continue
                new_label_idx = i + 1
                new_label_infos.append((new_label_idx, contours[i]))
            # 重新给前景文档赋值, reassign the foreground document
            fore_idx = new_label_idx + 1
            new_label_infos.append((fore_idx, label_infos_sorted[-1][1]))

            if too_small_contours > 0:
                print(file, too_small_contours)
            
            mask_new = np.zeros(img.shape[:2], dtype=np.uint8)
            new_label_infos_sorted = sorted(new_label_infos, key=lambda x: x[0])
            for label_idx, points in new_label_infos_sorted:
                points = np.asarray(points).astype(np.int32)
                cv2.fillPoly(mask_new, [points], label_idx)

            points_json[file] = foreground_quad.tolist()
            #print("img_path:", os.path.join(dst_dir, "img", file.replace(".jpg", ".png")) )
            #print("mask_path:", os.path.join(dst_dir, "mask", file.replace(".jpg", ".png")))
            shutil.copy(img_p, os.path.join(dst_dir, "img", file.replace(".jpg", ".png")))
            cv2.imwrite(os.path.join(dst_dir, "mask", file.replace(".jpg", ".png")), mask_new)
    json.dump(points_json, open(f"{dst_dir}/label_points.json", "w"), indent=4, ensure_ascii=False)

def rotate_img(root_dir, out_dir):
    # TODO 若旋转原始图片，清晰度会略微提升
    def save_img(json_path):
        label = json.load(open(json_path, "r"), encoding="utf-8")
        data_bytes = b64decode(label["imageData"])
        bytes_stream = BytesIO(data_bytes)
        image = Image.open(bytes_stream)
        
        new_img_name = os.path.split(json_path)[-1].replace(".json", ".png")
        new_img_path = os.path.join(out_dir, new_img_name)
        image.save(new_img_path)

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    for root, dirs, files in os.walk(root_dir):
        for file in tqdm(files):
            if file.endswith(".json"):
                json_path = os.path.join(root, file)
                save_img(json_path)
            
def split_data(root_dir):
    def split(img_dir, mask_dir, img_names, out_dir):
        for img_name in tqdm(img_names):
            img_path = os.path.join(img_dir, img_name)
            mask_path = os.path.join(mask_dir, img_name)
            shutil.copy2(img_path, os.path.join(out_dir, "img"))
            shutil.copy2(mask_path, os.path.join(out_dir, "mask"))

    img_dir = os.path.join(root_dir, "img")
    mask_dir = os.path.join(root_dir, "mask")
    

    if not os.path.exists(img_dir):
        os.mkdir(img_dir)
    if not os.path.exists(mask_dir):
        os.mkdir(mask_dir)
    
    
    img_name_list = os.listdir(img_dir)
    random.shuffle(img_name_list)
    train_num = int(len(img_name_list) * 0.75)
    
    img_names_train = img_name_list[:train_num]
    img_names_test = img_name_list[train_num:]

    split(img_dir, mask_dir, img_names_train, os.path.join(root_dir, "train"))
    split(img_dir, mask_dir, img_names_test, os.path.join(root_dir, "test"))

def resize_customdata(src_dir, dst_dir, json_path, json_save={}):
    # resize img and mask and points
    imgs_paths = glob.glob(src_dir + "/img/*png")
    json_label = json.load(open(json_path, "r"))
    for img_p in tqdm(imgs_paths):
        img = cv2.imread(img_p)
        img_n = os.path.basename(img_p)
        label_p = os.path.join(src_dir, "mask", img_n)
        mask = cv2.imread(label_p, cv2.IMREAD_GRAYSCALE)
        # 最大边缩放到1024 , scale the longest edge of the image to 1024
        max_size = max(img.shape[0], img.shape[1])
        scale = 1.0
        if max_size > 1024:
            scale = 1024 / max_size
            img = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, (int(mask.shape[1] * scale), int(mask.shape[0] * scale)), interpolation=cv2.INTER_NEAREST)
        
        # cv2.imwrite(os.path.join(dst_dir, "img", img_n), img)
        # cv2.imwrite(os.path.join(dst_dir, "mask", img_n), mask)

        # import pdb; pdb.set_trace()
        point_scale = np.asarray(json_label[img_n]) * scale
        json_save[img_n] = point_scale.tolist()
    json.dump(json_save, open(os.path.join(dst_dir, "label_points_resize.json"), "w"), indent=4, ensure_ascii=False)
    return json_save


if __name__ == "__main__":
    ori_dir = "/data2/xuzhen8/duan/data/2009"
    target_dir = "/data2/xuzhen8/duan/data/2009_rotate_format_v2"
    
    #ori_dir = "/data2/xuzhen8/duan/data/public"
    target_dir = "/data2/xuzhen8/duan/data/public_rotate_format"
    # To gain the mask labels form the original json label files
    print("generate_labele processing ....")
    genarate_label_from_ori(ori_dir, target_dir)
    # Split the dataset into training and test dataset
    print("split_data processing ....")
    split_data(target_dir)

    # optional operation for resizing the images and labels to the certain size
    '''
    json_save = resize_customdata("/data2/xuzhen8/duan/data/2009_rotate_format_v2/train",
                    "/data2/xuzhen8/duan/data/2009_rotate_format_v2/train_resize",
                    "/data2/xuzhen8/duan/data/2009_rotate_format_v2/label_points.json")
    resize_customdata("/data2/xuzhen8/duan/data/2009_rotate_format_v2/test",
                "/data2/xuzhen8/duan/data/2009_rotate_format_v2/test_resize",
                "/data2/xuzhen8/duan/data/2009_rotate_format_v2/label_points.json",
                json_save)
    '''
    
