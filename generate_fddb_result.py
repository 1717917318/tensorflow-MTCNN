# coding:utf-8
from __future__ import division
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
import time
import dlib
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

def ok(x1,y1,x2,y2,w,h):
    if x1>w or x1<0 or x2>w or x2<0 or y1>h or y1<0 or y2>h or y2<0:
        return False
    return True
def detectFaceOpenCVDnn(net, img, flag=False): # not sure is ok or not.
    # frameOpencvDnn = frame.copy()
    # frameHeight = frameOpencvDnn.shape[0]
    # frameWidth = frameOpencvDnn.shape[1]
    w = img.shape[1]
    h = img.shape[0]
    # flag = False
    # if w<h and w<1200 and h<1200: #or w/h>0.75:
    #     flag = True
    # if  flag == True:
    #     blob = cv2.dnn.blobFromImage(img, 1.0, (300,300), [104, 117, 123], False, False)
    # else :
    if flag != True:
        blob = cv2.dnn.blobFromImage(img, 1.0, (w, h), [104, 117, 123], False, False)
    else:
        blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), [104, 117, 123], False, False)

    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        x1 = int(detections[0, 0, i, 3] * w)
        y1 = int(detections[0, 0, i, 4] * h)
        x2 = int(detections[0, 0, i, 5] * w)
        y2 = int(detections[0, 0, i, 6] * h)
        if ok(x1,y1,x2,y2,w,h)== False:
            continue
        bboxes.append([x1, y1, x2, y2, confidence])
            #cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight/150)), 8)
    return bboxes
def detectFaceDlibMMOD(detector, img, inHeight=300, inWidth=0):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faceRects = detector(img, 0)
    bboxes = []
    for faceRect in faceRects:
        cvRect = [int(faceRect.rect.left() ), int(faceRect.rect.top() ),
                          int(faceRect.rect.right()), int(faceRect.rect.bottom() ) ]
        bboxes.append(cvRect)
    return img, bboxes

def detectFaceDlibHog(detector, img, inHeight=300, inWidth=0):

    #frameDlibHog = frame.copy()
    h = img.shape[0]
    w = img.shape[1]
    if not inWidth:
        inWidth = int((w / h)*inHeight)

    # scaleHeight = frameHeight / inHeight
    # scaleWidth = frameWidth / inWidth
    #
    # frameDlibHogSmall = cv2.resize(frameDlibHog, (inWidth, inHeight))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faceRects = detector(img, 0)
    #print(w, h, inWidth, inHeight)
    bboxes = []
    for faceRect in faceRects:

        cvRect = [int(faceRect.left()), int(faceRect.top()),
                  int(faceRect.right()), int(faceRect.bottom()) ]
        bboxes.append(cvRect)
        cv2.rectangle(img, (cvRect[0], cvRect[1]), (cvRect[2], cvRect[3]), (0, 255, 0),  2)
    return img, bboxes

def exist_both_box(box,boxes):
    exist_both = True

    '''裁剪的box和图片所有人脸box的iou值
    参数：
      box：裁剪的box,当box维度为4时表示box左上右下坐标，维度为5时，最后一维为box的置信度
      boxes：图片所有人脸box,[n,4]
    返回值：
      iou值，[n,]
    '''
    # box面积
    box_area = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
    # boxes面积,[n,]
    area = (boxes[ 2] - boxes[ 0] + 1) * (boxes[3] - boxes[ 1] + 1)
    # 重叠部分左上右下坐标
    xx1 = max(box[0], boxes[0])
    yy1 = max(box[1], boxes[1])
    xx2 = min(box[2], boxes[2])
    yy2 = min(box[3], boxes[3])

    # 重叠部分长宽
    w =  xx2 - xx1 + 1
    h =  yy2 - yy1 + 1

    if w<0 or h<0 :
        return True

    # 重叠部分面积
    inter = w * h
    common1 = inter / (box_area+ 1e-10)
    common2 = inter /( area + 1e-10  )
    if common1 >0.4 : #or common2 >0.5 :
        exist_both = False
    return exist_both

def del_common_box(boxes):

    refined_boxes = np.array( [0,0,0,0,0])

    for i in range( len (boxes ) ):
        cur_box = boxes[i]
        can_append = True
        for j in range( len(boxes) ):
            if j == i :
                continue
            cmp_box = boxes[j]
            can_append =  exist_both_box(cur_box, cmp_box)
            if not can_append:
                break
        if can_append:
            refined_boxes = np.vstack( (refined_boxes, cur_box) )


    return refined_boxes[1: len(refined_boxes)]


if __name__ == '__main__':

    data_dir = '../DATA/FDDB/originalPics' #验证集（测试）路径
    anno_file = 'fddb_val_min_r.txt' #验证集标签文件（包含路径）
    # anno_file = 'wider_val_min_r.txt'  # 验证集标签文件（包含路径）
    name = anno_file.split('.')[0]
    # if anno_file[0]=='f':
    anno_file = 'test/'+ anno_file
    all_methods = ['hog','mmod','dnn','haar','mtcnn']
    write_img = False

    haar_xml_file_pos = './FaceDetectionComparision/models/haarcascade_frontalface_default.xml'
    print("loading test images ...")
    image_info = get_image_info(anno_file)
    print("load test images done.")

    for detect_method in all_methods:

        print("current test method: "+detect_method)
        #detect_method = 'mmod'  # 'mmod' / 'hog' / 'dnn' / 'haar'或者 'mtcnn' 测试模型类型
        
        output_file = 'output/' + name +detect_method # 与标签文件对应的目录
        val_test_result_name = name + detect_method+"_test_result.txt"
        out_path = output_file
        # 生成预测框文件，每个框的格式都为x1, y1, x1+w, y1+h, possibility

        if (detect_method != 'dnn'):
            print("jump this method "+ detect_method)
            continue
        start_time = time.time()

        if detect_method == 'mtcnn':
    
            test_mode = "ONet"
            #thresh = [0.3, 0.1, 0.7]
            # thresh = [0.3, 0.1, 0.7] #网络的置信度阈值
            thresh = [0.1, 0.2, 0.3]  # 网络的置信度阈值
            min_face_size = 20
            stride = 2
    
    
            shuffle = False
            vis = False
            detectors = [None, None, None]
            # prefix is the model path
            prefix = ['model/PNet', 'model/RNet',
                      'model/ONet']
            epoch = [30, 22, 22]  # 原来onet训练14轮
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
    
            
    
            current_month = ''
            save_path = ''
            idx = 0
    
    
            if(not os.path.exists(output_file)):
                os.mkdir(output_file)
    
            output_file_name = os.path.join(output_file,val_test_result_name)
            fid = open(output_file_name,"w")
    
            for item in image_info:
                idx+=1
                image_file_name = os.path.join(data_dir, item[0], item[1],item[2],item[3],item[4]+'.jpg')
                if current_month != item[1]:
                    current_month = item[1]
                    print('current path:', current_month)
    
                # generate detection
                img = cv2.imread(image_file_name)
                boxes_c, landmarks= mtcnn_detector.detect(img)


                #f_name = item[1].split('.jpg')[0]
    
                #dets_file_name = os.path.join(save_path, f_name + '.txt')
                #fid =open (dets_file_name,'w')
                short_file_name = os.path.join(item[0],item[1],item[2],item[3],item[4])
                if boxes_c.shape[0] == 0 :
                    fid.write(short_file_name+ ' ')
                    fid.write(str(0) + '\n')
                    #fid.write('%f %f %f %f %f\n' % (0, 0, 0, 0, 0.99))
                    continue
    
                fid.write(short_file_name + ' ')
                fid.write(str(len(boxes_c)) )
                # if item[4] == 'img_3508' :
                #     print("")
                boxes_c = del_common_box(boxes_c)

                for box in boxes_c:
                    fid.write(' %d %d %d %d %f' % (
                    int(box[0]), int(box[1]), int(box[2]), int(box[3]), box[4])) # x,y,x+w,y+h,possibility
                fid.write("\n")
    
                for i in range(boxes_c.shape[0]):
                    bbox = boxes_c[i, :4]
                    score = boxes_c[i, 4]
                    corpbbox = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
                    # 画人脸框
                    if write_img:
                        cv2.rectangle(img, (corpbbox[0], corpbbox[1]),
                                      (corpbbox[2], corpbbox[3]), (255, 0, 0), 1)
                        # 判别为人脸的置信度
                        cv2.putText(img, '{:.2f}'.format(score),
                                    (corpbbox[0], corpbbox[1] - 2),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
                if write_img:
                    # 画关键点
                    for i in range(landmarks.shape[0]):
                        for j in range(len(landmarks[i]) // 2):
                            cv2.circle(img, (int(landmarks[i][2 * j]), int(int(landmarks[i][2 * j + 1]))), 2, (0, 0, 255))
                            #fid.close()
                    output_path = os.path.join(output_file, item[0])
                    if (not os.path.exists(output_path)):
                        os.mkdir(output_path)
                    cv2.imwrite(output_path + "/" + item[4] + '.jpg', img)
                print(str(idx)+" pictures has done.")
                if idx % 10 == 0:
                    print(idx)
            fid.close()
        elif detect_method =='haar':
            
    
            current_month = ''
            save_path = ''
            idx = 0
    
            if (not os.path.exists(output_file)):
                os.mkdir(output_file)
    
            output_file_name = os.path.join(output_file, val_test_result_name)
            fid = open(output_file_name, "w")
    
            for item in image_info:
                idx += 1
                image_file_name = os.path.join(data_dir, item[0], item[1],item[2],item[3],item[4]+'.jpg')
                if current_month != item[1]:
                    current_month = item[1]
                    print('current path:', current_month)
    
                # generate detection
                img = cv2.imread(image_file_name)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                detector = cv2.CascadeClassifier(haar_xml_file_pos)
                faces = detector.detectMultiScale(gray, 1.3, 5)
    
    
                short_file_name = os.path.join(item[0], item[1],item[2],item[3],item[4])
                if len(faces) == 0:
                    fid.write(short_file_name + ' ')
                    fid.write(str(0) + '\n')
                    # fid.write('%f %f %f %f %f\n' % (0, 0, 0, 0, 0.99))
                    continue
    
                fid.write(short_file_name + ' ')
                fid.write(str(int(len(faces)) ))
    
                for (x,y,w,h) in faces:
                    fid.write(' %d %d %d %d 1' % (
                        int(x), int(y), int(x+w), int(y+h) ) )  # x,y,x+w,y+h,possibility
                    if write_img:
                        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                fid.write("\n")
    
                #cv2.imshow('im', img)
                #k = cv2.waitKey(0) & 0xFF
                #if k == 27:
                if write_img:
                    output_path = os.path.join(output_file,item[0])
                    if (not os.path.exists(output_path) ):
                        os.mkdir(output_path)
                    cv2.imwrite(output_path+"/"+item[4]+'.jpg', img)
    
    
                if idx % 10 == 0:
                    print(idx)
            fid.close()
            #cv2.destroyAllWindows()
        elif detect_method == 'dnn':
            
    
            current_month = ''
            save_path = ''
            idx = 0
    
            if (not os.path.exists(output_file)):
                os.mkdir(output_file)
    
            output_file_name = os.path.join(output_file, val_test_result_name)
            fid = open(output_file_name, "w")
    
    
    
    
            for item in image_info:
                #net import
                # OpenCV DNN supports 2 networks.
                # 1. FP16 version of the original caffe implementation ( 5.4 MB )
                # 2. 8 bit Quantized version using Tensorflow ( 2.7 MB )
                DNN = "TF"
                prefix = 'FaceDetectionComparision/'
                if DNN == "CAFFE":
                    modelFile = prefix + "models/res10_300x300_ssd_iter_140000_fp16.caffemodel"
                    configFile = prefix + "models/deploy.prototxt"
                    net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
                else:
                    modelFile = prefix + "models/opencv_face_detector_uint8.pb"
                    configFile = prefix + "models/opencv_face_detector.pbtxt"
                    net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)
                    net2 = cv2.dnn.readNetFromTensorflow(modelFile, configFile)

                #end
    
                idx += 1
                print("idx: "+str(idx))
                image_file_name = os.path.join(data_dir, item[0], item[1],item[2],item[3],item[4]+'.jpg')
                if current_month != item[1]:
                    current_month = item[1]
                    print('current path:', current_month)
    
                # generate detection
                img = cv2.imread(image_file_name)
                #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    
                bboxes = detectFaceOpenCVDnn(net, img)
                # bboxes = del_common_box(bboxes)
                bboxes2 = detectFaceOpenCVDnn(net2, img, True)
                # bboxes2 = del_common_box(bboxes2)
                if (len(bboxes2) > len(bboxes)):
                    bboxes = bboxes2
                #cv2.putText(outOpencvDnn, label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3, cv2.LINE_AA)
                #cv2.imshow("Face Detection Comparison", outOpencvDnn)
    
                short_file_name = os.path.join(item[0], item[1],item[2],item[3],item[4])
                if len(bboxes) == 0:
                    fid.write(short_file_name + ' ')
                    fid.write(str(0) + '\n')
                    # fid.write('%f %f %f %f %f\n' % (0, 0, 0, 0, 0.99))
                    continue
    
                fid.write(short_file_name + ' ')
                fid.write(str(len(bboxes)))
    
                for (x1, y1, x2, y2, conf) in bboxes:

                    fid.write(' %d %d %d %d %f' % (
                        int(x1), int(y1), int(x2), int(y2),conf) )  # x,y,x+w,y+h,possibility
                    if write_img:
                        cv2.rectangle(img, (int(x1), int(y1) ), ( int(x2), int(y2) ), (255, 0, 0), 2)
                fid.write("\n")
    
                #cv2.imshow('im', img)
                #k = cv2.waitKey(0) & 0xFF
                #if k == 27:
                if write_img:
                    output_path = os.path.join(output_file, item[0])
                    if (not os.path.exists(output_path)):
                       os.mkdir(output_path)
                    cv2.imwrite(output_path + "/" + item[4] + '.jpg', img)
    
                if idx % 10 == 0:
                    print(idx)
            fid.close()
            #cv2.destroyAllWindows()
            #pass
        elif detect_method == 'hog':
            
    
            current_month = ''
            save_path = ''
            idx = 0
    
            if (not os.path.exists(output_file)):
                os.mkdir(output_file)
    
            output_file_name = os.path.join(output_file, val_test_result_name)
            fid = open(output_file_name, "w")
    
            hogFaceDetector = dlib.get_frontal_face_detector()
    
            for item in image_info:
                # net import
                # OpenCV DNN supports 2 networks.
                # 1. FP16 version of the original caffe implementation ( 5.4 MB )
                # 2. 8 bit Quantized version using Tensorflow ( 2.7 MB )
    
    
                conf_threshold = 1
                # end
    
                idx += 1
                print("idx: " + str(idx))
                image_file_name = os.path.join(data_dir, item[0], item[1],item[2],item[3],item[4]+'.jpg')
                if current_month != item[1]:
                    current_month = item[1]
                    print('current path:', current_month)
    
                # generate detection
                img = cv2.imread(image_file_name)
                # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
                img, bboxes = detectFaceDlibHog(hogFaceDetector, img)
    
                # cv2.putText(outOpencvDnn, label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3, cv2.LINE_AA)
                # cv2.imshow("Face Detection Comparison", outOpencvDnn)
    
                short_file_name = os.path.join(item[0], item[1],item[2],item[3],item[4])
                if len(bboxes) == 0:
                    fid.write(short_file_name + ' ')
                    fid.write(str(0) + '\n')
                    # fid.write('%f %f %f %f %f\n' % (0, 0, 0, 0, 0.99))
                    continue
    
                fid.write(short_file_name + ' ')
                fid.write(str(len(bboxes)))
    
                for (x1, y1, x2, y2) in bboxes:
                    fid.write(' %d %d %d %d %f' % (
                        int(x1), int(y1), int(x2), int(y2), conf_threshold))  # x,y,x+w,y+h,possibility
                    if write_img:
                        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                fid.write("\n")
    
                # cv2.imshow('im', img)
                # k = cv2.waitKey(0) & 0xFF
                # if k == 27:
                if write_img:
                    output_path = os.path.join(output_file, item[0])
                    if (not os.path.exists(output_path)):
                       os.mkdir(output_path)
                    cv2.imwrite(output_path + "/" + item[4] + '.jpg', img)
    
                if idx % 10 == 0:
                    print(idx)
            fid.close()
        elif detect_method =='mmod':
            
    
            current_month = ''
            save_path = ''
            idx = 0
    
            if (not os.path.exists(output_file)):
                os.mkdir(output_file)
    
            output_file_name = os.path.join(output_file, val_test_result_name)
            fid = open(output_file_name, "w")
            prefix = 'FaceDetectionComparision/'
            dnnFaceDetector = dlib.cnn_face_detection_model_v1(prefix+"models/mmod_human_face_detector.dat")
    
            for item in image_info:
    
                conf_threshold = 1
                # end
    
                idx += 1
                print("idx: " + str(idx))
                image_file_name = os.path.join(data_dir, item[0], item[1],item[2],item[3],item[4]+'.jpg')
                if current_month != item[1]:
                    current_month = item[1]
                    print('current path:', current_month)
    
                # generate detection
                img = cv2.imread(image_file_name)
                # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img, bboxes = detectFaceDlibMMOD(dnnFaceDetector, img)
    
    
                # cv2.putText(outOpencvDnn, label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3, cv2.LINE_AA)
                # cv2.imshow("Face Detection Comparison", outOpencvDnn)
    
                short_file_name = os.path.join(item[0], item[1],item[2],item[3],item[4])
                if len(bboxes) == 0:
                    fid.write(short_file_name + ' ')
                    fid.write(str(0) + '\n')
                    # fid.write('%f %f %f %f %f\n' % (0, 0, 0, 0, 0.99))
                    continue
    
                fid.write(short_file_name + ' ')
                fid.write(str(len(bboxes)))
    
                for (x1, y1, x2, y2) in bboxes:
                    fid.write(' %d %d %d %d %f' % (
                        int(x1), int(y1), int(x2), int(y2), conf_threshold))  # x,y,x+w,y+h,possibility
                    if write_img:
                        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                fid.write("\n")
    
                # cv2.imshow('im', img)
                # k = cv2.waitKey(0) & 0xFF
                # if k == 27:
                if write_img:
                    output_path = os.path.join(output_file, item[0])
                    if (not os.path.exists(output_path)):
                       os.mkdir(output_path)
                    cv2.imwrite(output_path + "/" + item[4] + '.jpg', img)
    
                if idx % 10 == 0:
                    print(idx)
            fid.close()
        else :
            print("Model Wrong!")

        print(detect_method+" method test done")
        print("---It costs %s seconds ---\n" % (time.time() - start_time))
    print("Generate all "+anno_file+" images data!")
    print("Saved to "+output_file_name)
