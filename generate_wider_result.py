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

    data_dir = 'data/WIDER_val/images' #验证集（测试）路径
    anno_file = 'wider_face_val_small.txt' #验证集标签文件（包含路径）
    name = anno_file.split('.')[0]


    detect_method = 'mtcnn'  # 'haar'或者 'mtcnn' 测试模型类型
    haar_xml_file_pos = '/Users/zjt/Documents/Develop/OpenCV_haarcascade/frontalFace10/haarcascade_frontalface_default.xml'
    output_file = 'output/' + name +detect_method # 与标签文件对应的目录
    val_test_result_name = name + detect_method+"_test_result.txt"
    out_path = output_file
    if detect_method == 'mtcnn':

        test_mode = "PNet"
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

            for i in range(boxes_c.shape[0]):
                bbox = boxes_c[i, :4]
                score = boxes_c[i, 4]
                corpbbox = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
                # 画人脸框
                cv2.rectangle(img, (corpbbox[0], corpbbox[1]),
                              (corpbbox[2], corpbbox[3]), (255, 0, 0), 1)
                # 判别为人脸的置信度
                cv2.putText(img, '{:.2f}'.format(score),
                            (corpbbox[0], corpbbox[1] - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            # 画关键点
            for i in range(landmarks.shape[0]):
                for j in range(len(landmarks[i]) // 2):
                    cv2.circle(img, (int(landmarks[i][2 * j]), int(int(landmarks[i][2 * j + 1]))), 2, (0, 0, 255))
                    #fid.close()
            output_path = os.path.join(output_file, item[0])
            if (not os.path.exists(output_path)):
                os.mkdir(output_path)
            cv2.imwrite(output_path + "/" + item[1] + '.jpg', img)
            print(idx+" pictures has done.")
            if idx % 10 == 0:
                print(idx)
        fid.close()
    elif detect_method =='haar':

        image_info = get_image_info(anno_file)

        current_event = ''
        save_path = ''
        idx = 0

        if (not os.path.exists(output_file)):
            os.mkdir(output_file)

        output_file_name = os.path.join(output_file, val_test_result_name)
        fid = open(output_file_name, "w")

        for item in image_info:
            idx += 1
            image_file_name = os.path.join(data_dir, item[0], item[1])
            if current_event != item[0]:
                current_event = item[0]
                print('current path:', current_event)

            # generate detection
            img = cv2.imread(image_file_name)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            detector = cv2.CascadeClassifier(haar_xml_file_pos)
            faces = detector.detectMultiScale(gray, 1.3, 5)




            # f_name = item[1].split('.jpg')[0]

            # dets_file_name = os.path.join(save_path, f_name + '.txt')
            # fid =open (dets_file_name,'w')
            short_file_name = os.path.join(item[0], item[1])
            if len(faces) == 0:
                fid.write(short_file_name + ' ')
                fid.write(str(0) + '\n')
                # fid.write('%f %f %f %f %f\n' % (0, 0, 0, 0, 0.99))
                continue

            fid.write(short_file_name + ' ')
            fid.write(str(len(faces)//4))

            for (x,y,w,h) in faces:
                fid.write(' %d %d %d %d' % (
                    int(x), int(y), int(x+w), int(y+h) ) )  # x,y,x+w,y+h,possibility
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            fid.write("\n")

            #cv2.imshow('im', img)
            #k = cv2.waitKey(0) & 0xFF
            #if k == 27:
            output_path = os.path.join(output_file,item[0])
            if (not os.path.exists(output_path) ):
                os.mkdir(output_path)
            cv2.imwrite(output_path+"/"+item[1]+'.jpg', img)


            # fid.close()
            if idx % 10 == 0:
                print(idx)
        fid.close()
        #cv2.destroyAllWindows()

    else :
        print("Model Wrong!")

    print("Generate all "+anno_file+" images data!")
    print("Saved to "+output_file_name)
