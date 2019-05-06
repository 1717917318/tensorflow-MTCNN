from __future__ import division
import cv2
import time
import sys


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
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * w)
            y1 = int(detections[0, 0, i, 4] * h)
            x2 = int(detections[0, 0, i, 5] * w)
            y2 = int(detections[0, 0, i, 6] * h)
            if ok(x1,y1,x2,y2,w,h)== False:
                continue
            bboxes.append([x1, y1, x2, y2])
            #cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight/150)), 8)
    return bboxes

if __name__ == "__main__" :

    # OpenCV DNN supports 2 networks.
    # 1. FP16 version of the original caffe implementation ( 5.4 MB )
    # 2. 8 bit Quantized version using Tensorflow ( 2.7 MB )
    DNN = "TF"
    if DNN == "CAFFE":
        modelFile = "models/res10_300x300_ssd_iter_140000_fp16.caffemodel"
        configFile = "models/deploy.prototxt"
        net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
    else:
        modelFile = "models/opencv_face_detector_uint8.pb"
        configFile = "models/opencv_face_detector.pbtxt"
        net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)
        net2 = cv2.dnn.readNetFromTensorflow(modelFile, configFile)
        #net3 = cv2.dnn.readNetFromTensorflow(modelFile, configFile)

    conf_threshold = 0.7
    for i in range(1):
        # generate detection
        img = cv2.imread("../data/WIDER_val/images/1--Handshaking/1_Handshaking_Handshaking_1_766.jpg")
        #img = cv2.imread("../data/WIDER_val/images/10--People_Marching/10_People_Marching_People_Marching_2_935.jpg")
        #img = cv2.imread("../data/WIDER_val/images/10--People_Marching/10_People_Marching_People_Marching_10_People_Marching_People_Marching_10_552.jpg")
        #img = cv2.imread("../data/WIDER_val/images/11--Meeting/11_Meeting_Meeting_11_Meeting_Meeting_11_529.jpg")
        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        bboxes = detectFaceOpenCVDnn(net, img)

        # img2=cv2.resize(img,(img.shape[0]//2,img.shape[1]//2))
        bboxes2 = detectFaceOpenCVDnn(net2, img,True)
        if (len(bboxes2) > len(bboxes)):
            bboxes = bboxes2

        # img3 = cv2.resize(img, (int(img2.shape[0] * 0.5), int(img2.shape[1] * 0.5)))
        # bboxes3 = detectFaceOpenCVDnn(net3, img3)
        # cv2.imwrite("../output/single/" + 'img3.jpg', img3)
        # if (len(bboxes3) > len(bboxes)):
        #     bboxes = bboxes2

        # cv2.putText(outOpencvDnn, label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3, cv2.LINE_AA)
        # cv2.imshow("Face Detection Comparison", outOpencvDnn)
        fid = open("../output/single/dnn_test.txt", "w")
        short_file_name =  "jpg_name"
        if len(bboxes) == 0:
            fid.write(short_file_name + ' ')
            fid.write(str(0) + '\n')
            # fid.write('%f %f %f %f %f\n' % (0, 0, 0, 0, 0.99))


        fid.write(short_file_name + ' ')
        fid.write(str(len(bboxes) // 4))

        for (x1, y1, x2, y2) in bboxes:
            fid.write(' %d %d %d %d' % (
                int(x1), int(y1), int(x2), int(y2)))  # x,y,x+w,y+h,possibility
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        fid.write("\n")

        # cv2.imshow('im', img)
        # k = cv2.waitKey(0) & 0xFF
        # if k == 27:

        # output_path = os.path.join(output_file, item[0])
        # if (not os.path.exists(output_path)):
        #    os.mkdir(output_path)
        cv2.imwrite("../output/single/" + 'single_dnn.jpg', img)

        print("here")
    ##2
    img = cv2.imread("../data/WIDER_val/images/10--People_Marching/10_People_Marching_People_Marching_2_171.jpg")
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    bboxes = detectFaceOpenCVDnn(net, img)
    # cv2.putText(outOpencvDnn, label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3, cv2.LINE_AA)
    # cv2.imshow("Face Detection Comparison", outOpencvDnn)
    #fid = open("../output/single/dnn_test.txt", "w")
    short_file_name = "jpg_name"
    if len(bboxes) == 0:
        fid.write(short_file_name + ' ')
        fid.write(str(0) + '\n')
        # fid.write('%f %f %f %f %f\n' % (0, 0, 0, 0, 0.99))

    fid.write(short_file_name + ' ')
    fid.write(str(len(bboxes) // 4))

    for (x1, y1, x2, y2) in bboxes:
        fid.write(' %d %d %d %d' % (
            int(x1), int(y1), int(x2), int(y2)))  # x,y,x+w,y+h,possibility
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
    fid.write("\n")
    cv2.imwrite("../output/single/" + 'single_dnn_two.jpg', img)
    ##2
    print("here2")



    fid.close()
    cv2.destroyAllWindows()
