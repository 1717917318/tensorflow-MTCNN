import  os
import numpy as np
import math


class BBox:
    # 人脸的box
    def __init__(self, box):
        self.left = box[0]
        self.top = box[1]
        self.right = box[2]
        self.bottom = box[3]

        self.x = box[0]
        self.y = box[1]
        self.w = box[2] - box[0]
        self.h = box[3] - box[1]

    def project(self, point):
        '''将关键点的绝对值转换为相对于左上角坐标偏移并归一化
        参数：
          point：某一关键点坐标(x,y)
        返回值：
          处理后偏移
        '''
        x = (point[0] - self.x) / self.w
        y = (point[1] - self.y) / self.h
        return np.asarray([x, y])

    def reproject(self, point):
        '''将关键点的相对值转换为绝对值，与project相反
        参数：
          point:某一关键点的相对归一化坐标
        返回值：
          处理后的绝对坐标
        '''
        x = self.x + self.w * point[0]
        y = self.y + self.h * point[1]
        return np.asarray([x, y])

    def reprojectLandmark(self, landmark):
        '''对所有关键点进行reproject操作'''
        p = np.zeros((len(landmark), 2))
        for i in range(len(landmark)):
            p[i] = self.reproject(landmark[i])
        return p

    def projectLandmark(self, landmark):
        '''对所有关键点进行project操作'''
        p = np.zeros((len(landmark), 2))
        for i in range(len(landmark)):
            p[i] = self.project(landmark[i])
        return p



def getDataFromTxt(txt, data_path, with_landmark=False):
    '''获取txt中的图像路径，人脸box
    参数：
      txt：数据txt文件
      data_path:数据存储目录
      with_landmark:是否留有关键点
    返回值：
      result包含(图像路径，人脸box，关键点)
    '''
    with open(txt, 'r') as f:
        lines = f.readlines()
    result = np.array([])
    i = 0
    while i< len(lines):
    #for line in lines:
        img_path = lines[i]
        i = i + 1
        cnt =  int (lines[i])
        #img_path = np.append(img_path, cnt)
        boxes = np.array([],dtype='int')
        for j in range(cnt):
            i = i + 1
            boxline = lines[i]
            boxline = boxline[:-1]
            components = boxline.split(' ')
            components.remove('')
            # boxline[5] = boxline[6]
            # del boxline[-1]

            [max_r,min_r,angle,center_x,center_y,conf] = [float(_) for _ in components]
            x1 = center_x + min_r * math.cos(angle)
            x2 = center_x - min_r * math.cos(angle)
            y1 = center_y + min_r * math.sin(angle)
            y2 = center_y - min_r * math.sin(angle)
            x = int(min(x1,x2))
            x_w = int (max(x1,x2))
            y = int (min(y1,y2))
            y_h = int (max(y1,y2))
            box =np.array( [x,y,x_w,y_h]  )
            #box = list(map(int, box))
            boxes = np.append(boxes, box )
        i = i + 1

        if not with_landmark:
            single_pic = np.append(img_path,boxes)
            result = np.append(result, single_pic )
            continue

    return result



if __name__ == '__main__' :

    merged_fddb_file = '/Users/zjt/MEGAsync/毕设/基于深度学习的人脸检测算法设计与实现/1.CNN实现/参考实现/DATA/FDDB/FDDB-folds/FDDB-fold-all-ellipseList.txt'
    output_fddb_val_file = 'fddb_val.txt'
    result = getDataFromTxt(merged_fddb_file,'.')
    output_file = open(output_fddb_val_file,'w')
    for i in range(len(result) ):
       element = result[i]
       if (i>0) and ('/' in element) :
           output_file.write('\n')
       if element[-1] == '\n':
           element = element [0:-1] 
       output_file.write(str(element) )
       if i<(len(result)-1) and ('/' not in result[i+1] ):
        output_file.write(' ')
        
    output_file.close()
    print("done")