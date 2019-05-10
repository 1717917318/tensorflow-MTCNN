import os
import cv2
import math

def draw_ellipse(img,c_x,c_y,max_r,min_r,angle): # angle in  radian

    pi = math.pi
    angle = angle / pi * 180
    if angle < 0:
        angle += 360
    cv2.ellipse(img, (int(c_x), int(c_y)), (int(max_r), int(min_r)), angle, 0, 360, (255, 0, 0),1)  # [240,240,240], thickness = 1


if __name__ == '__main__':

    #fid =  open('','w')
    img = cv2.imread("../../DATA/FDDB/originalPics/2002/08/11/big/img_591.jpg")
    x,y,x_w,y_h= [184,76,355,247]
    cv2.rectangle(img,(x,y),(x_w,y_h),(255,0,222), 2 )

    #max_r,min_r,angle,c_x,c_y= [123.583300,85.549500,1.265839,269.693400,161.781200]
    max_r = 123.583300
    min_r = 85.549500
    angle = 1.265839




    c_x  = 269.693400
    c_y = 161.781200

    cv2.ellipse(img, (200, 200), (80, 50), 0,           0,  360 , (0, 255, 0), 1) # green
    cv2.imwrite('draw_sing.jpg',img)

