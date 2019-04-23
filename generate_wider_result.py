# coding:utf-8
import sys
import numpy as np

sys.path.append("..")
import argparse
from train.model import P_Net, R_Net, O_Net
from preprocess.loader import TestLoader
from detection.detector import Detector
from detection.fcn_detector import FcnDetector
from detection.MtcnnDetector import MtcnnDetector
import cv2
import os



def read_gt_bbox(raw_list):
    list_len = len(raw_list)
    bbox_num = (list_len - 1) // 4
    idx = 1
    bboxes = np.zeros((bbox_num, 4), dtype=int)
    for i in range(4):
        for j in range(bbox_num):
            bboxes[j][i] = int(raw_list[idx])
            idx += 1
    return bboxes


def get_image_info(anno_file):
    f = open(anno_file, 'r')
    image_info = []
    for line in f:
        ct_list = line.strip().split(' ')
        path = ct_list[0]

        path_list = path.split('\\')
        event = path_list[0]
        name = path_list[1]
        # print(event, name )
        bboxes = read_gt_bbox(ct_list)
        image_info.append([event, name, bboxes])
    print('total number of images in validation set: ', len(image_info))
    return image_info

if __name__ == '__main__':

    data_dir = 'data/WIDER_val/images'
    anno_file = 'wider_face_val_small.txt'
    name = anno_file.split('.')[0]
    output_file = 'output/'+name
    val_test_result_name=name+"_test_result.txt"

    test_mode = "ONet"
    thresh = [0.3, 0.1, 0.7]
    min_face_size = 20
    stride = 2

    shuffle = False
    vis = False
    detectors = [None, None, None]
    # prefix is the model path
    prefix = ['model/PNet', 'model/RNet',
              'model/ONet']
    epoch = [30, 22, 14]
    batch_size = [2048, 256, 16]
    #model_path = ['%s-%s' % (x, y) for x, y in zip(prefix, epoch)]
    model_path = prefix

    # load pnet model
    #if slide_window:
    #    PNet = Detector(P_Net, 12, batch_size[0], model_path[0])
    #else:
    PNet = FcnDetector(P_Net, model_path[0])
    detectors[0] = PNet

    # load rnet model
    if test_mode in ["RNet", "ONet"]:
        RNet = Detector(R_Net, 24, batch_size[1], model_path[1])
        detectors[1] = RNet

    # load onet model
    if test_mode == "ONet":
        ONet = Detector(O_Net, 48, batch_size[2], model_path[2])
        detectors[2] = ONet

    mtcnn_detector = MtcnnDetector(detectors=detectors, min_face_size=min_face_size,
                                   stride=stride, threshold=thresh)

    image_info = get_image_info(anno_file)

    current_event = ''
    save_path = ''
    idx = 0


    if(not os.path.exists(output_file)):
        os.mkdir(output_file)

    output_file_name = os.path.join(output_file,val_test_result_name)
    fid = open(output_file_name,"w")

    for item in image_info:
        idx+=1
        image_file_name = os.path.join(data_dir, item[0], item[1])
        if current_event != item[0]:
            current_event = item[0]
            print('current path:', current_event)

        # generate detection
        img = cv2.imread(image_file_name)
        boxes_c, landmarks= mtcnn_detector.detect(img)

        #f_name = item[1].split('.jpg')[0]

        #dets_file_name = os.path.join(save_path, f_name + '.txt')
        #fid =open (dets_file_name,'w')
        short_file_name=os.path.join(item[0],item[1])
        if boxes_c.shape[0] == 0 :
            fid.write(short_file_name+ ' ')
            fid.write(str(0) + '\n')
            #fid.write('%f %f %f %f %f\n' % (0, 0, 0, 0, 0.99))
            continue

        fid.write(short_file_name + ' ')
        fid.write(str(len(boxes_c)) )

        for box in boxes_c:
            fid.write(' %d %d %d %d %f' % (
            int(box[0]), int(box[1]), int(box[2]), int(box[3]), box[4])) # x,y,x+w,y+h,possibility
        fid.write("\n")

        #fid.close()
        if idx % 10 == 0:
            print(idx)
    fid.close()
    print("Generate all "+anno_file+" images data!")
    print("Saved to "+output_file_name)
