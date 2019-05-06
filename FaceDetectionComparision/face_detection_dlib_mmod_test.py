from __future__ import division
import cv2
import dlib
import time
import sys

def detectFaceDlibMMOD(detector, img, inHeight=300, inWidth=0):

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faceRects = detector(img, 0)

    # print(frameWidth, frameHeight, inWidth, inHeight)
    bboxes = []
    for faceRect in faceRects:
        cvRect = [int(faceRect.rect.left() ), int(faceRect.rect.top() ),
                  int(faceRect.rect.right()), int(faceRect.rect.bottom() ) ]
        bboxes.append(cvRect)
        #cv2.rectangle(img, (cvRect[0], cvRect[1]), (cvRect[2], cvRect[3]), (0, 255, 0), int(round(frameHeight/150)), 4)
    return img, bboxes

if __name__ == "__main__" :


    dnnFaceDetector = dlib.cnn_face_detection_model_v1("models/mmod_human_face_detector.dat")

    conf_threshold = 1
    fid = open("../output/single/mmod_test.txt", "w")

    for i in range(1):
        # generate detection
        img = cv2.imread("../data/WIDER_val/images/10--People_Marching/10_People_Marching_People_Marching_2_34.jpg")
        # img = cv2.imread("../data/WIDER_val/images/10--People_Marching/10_People_Marching_People_Marching_2_36.jpg")

        # img = cv2.imread("../data/WIDER_val/images/10--People_Marching/10_People_Marching_People_Marching_2_171.jpg")        # img = cv2.imread("../data/WIDER_val/images/1--Handshaking/1_Handshaking_Handshaking_1_766.jpg")
        # img = cv2.imread("../data/WIDER_val/images/10--People_Marching/10_People_Marching_People_Marching_2_935.jpg")
        # img = cv2.imread("../data/WIDER_val/images/10--People_Marching/10_People_Marching_People_Marching_10_People_Marching_People_Marching_10_552.jpg")
        # img = cv2.imread("../data/WIDER_val/images/11--Meeting/11_Meeting_Meeting_11_Meeting_Meeting_11_529.jpg")
        outDlibMMOD, bboxes = detectFaceDlibMMOD(dnnFaceDetector, img)
        #cv2.imshow("Face Detection Comparison", outDlibMMOD)

        short_file_name =  "jpg_name"
        if len(bboxes) == 0:
            fid.write(short_file_name + ' ')
            fid.write(str(0) + '\n')
            continue


        fid.write(short_file_name + ' ')
        fid.write(str(len(bboxes) ) )

        for (x1, y1, x2, y2) in bboxes:
            fid.write(' %d %d %d %d %f' % (
                int(x1), int(y1), int(x2), int(y2),conf_threshold) )  # x,y,x+w,y+h,possibility
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        fid.write("\n")

        cv2.imwrite("../output/single/" + 'single_mmod.jpg', img)
        #cv2.imshow('im', img)

        print("here")
    ##2
    #img = cv2.imread("../data/WIDER_val/images/10--People_Marching/10_People_Marching_People_Marching_2_171.jpg")
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    fid.close()
    #cv2.destroyAllWindows()