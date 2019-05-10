# coding:utf-8
from __future__ import division
import sys
import numpy as np
import time
import os
import cv2

sys.path.append("..")
#todo
#



def read_gt_bbox(raw_list):
    list_len = len(raw_list)
    bbox_num = (list_len - 1) // 4
    idx = 1
    bboxes = np.zeros((bbox_num, 4), dtype=int)
    for i in range(4):
        for j in range(bbox_num):
            bboxes[j][i] = int(raw_list[idx] )
            idx += 1
    return bboxes


def get_image_info(anno_file):
    f = open(anno_file, 'r')
    image_info = []
    for line in f:
        ct_list = line.strip().split(' ')
        path = ct_list[0]

        path_list = path.split('/')
        year = path_list[0]
        month = path_list[1]
        day = path_list[2]

        # print(event, name )
        bboxes = read_gt_bbox(ct_list)
        image_info.append([year,month,day,path_list[3],path_list[4], bboxes])
    print('total number of images in validation set: ', len(image_info))
    return image_info

def get_boxes(item):
    # faces = []
    # for i in range(5,len(item)-4,4):
    #     temp = [item[]]
    return item[5]

if __name__ == '__main__':

    data_dir = '../../DATA/FDDB/originalPics' #验证集（测试）路径
    anno_file = 'fddb_val_max_r_little.txt' #验证集标签文件（包含路径）
    name = anno_file.split('.')[0]
    write_img = True

    print("loading test images ...")
    image_info = get_image_info(anno_file)
    print("load test images done.")

    output_path = '../output/' + name   # 与标签文件对应的目录
    #out_path = output_path
    # 生成预测框文件，每个框的格式都为x1, y1, x1+w, y1+h, possibility
    start_time = time.time()

    current_month = ''
    save_path = ''
    idx = 0

    if (not os.path.exists(output_path)):
        os.mkdir(output_path)

    for item in image_info:
        idx += 1
        image_file_name = os.path.join(data_dir, item[0], item[1], item[2], item[3], item[4] + '.jpg')
        if current_month != item[1]:
            current_month = item[1]
            print('current path:', current_month)

        # generate detection
        img = cv2.imread(image_file_name)


        short_file_name = os.path.join(item[0], item[1])


        faces = get_boxes(item)

        for (x, y, x_w, y_h) in faces:
                cv2.rectangle(img, (x, y), (x_w, y_h), (255, 0, 0), 2)

        cv2.rectangle(img, (1,300 ), (201, 500), (255, 255, 255), 2)
        # cv2.imshow('im', img)
        # k = cv2.waitKey(0) & 0xFF
        # if k == 27:
        if write_img:

            output_path2 = os.path.join(output_path, item[0])
            if (not os.path.exists(output_path2)):
                os.mkdir(output_path2)
            cv2.imwrite(output_path2 + "/" + item[4] + '.jpg', img)

        if idx % 10 == 0:
            print(idx)