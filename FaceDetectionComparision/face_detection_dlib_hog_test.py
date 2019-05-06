from __future__ import division
import cv2
import dlib
import time
import sys

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

if __name__ == "__main__" :

    hogFaceDetector = dlib.get_frontal_face_detector()
    conf_threshold = 1
    fid = open("../output/single/hog_test.txt", "w")

    for i in range(1):
        # generate detection
        #img = cv2.imread("../data/WIDER_val/images/1--Handshaking/1_Handshaking_Handshaking_1_766.jpg")
        img = cv2.imread("../data/WIDER_val/images/10--People_Marching/10_People_Marching_People_Marching_2_935.jpg")
        #img = cv2.imread("../data/WIDER_val/images/10--People_Marching/10_People_Marching_People_Marching_10_People_Marching_People_Marching_10_552.jpg")
        #img = cv2.imread("../data/WIDER_val/images/11--Meeting/11_Meeting_Meeting_11_Meeting_Meeting_11_529.jpg")

        img, bboxes = detectFaceDlibHog(hogFaceDetector, img)
        #cv2.putText(outDlibHog, label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3, cv2.LINE_AA)
        cv2.imshow("Face Detection Comparison", img)

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

        cv2.imwrite("../output/single/" + 'single_hog.jpg', img)
        #cv2.imshow('im', img)

        print("here")
    ##2
    img = cv2.imread("../data/WIDER_val/images/10--People_Marching/10_People_Marching_People_Marching_2_171.jpg")
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    fid.close()
    cv2.destroyAllWindows()
