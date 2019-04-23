import  os
import numpy as np
import cv2
import matplotlib.pyplot as plt
#from sklearn.metrics import  auc  ###计算roc和auc
from preprocess.utils import IOU

# def get_TP():
#     TP = 0
#     return TP
#
# def get_FP():
#     FP = 0
#     return FP

def read_gt_bbox(raw_list):
    bbox_num = len(raw_list) //4
    idx = 0
    bboxes = np.zeros( (bbox_num,4), dtype=int)
    for i in range(4):
        for j in range(bbox_num):
            bboxes[j][i] = int (raw_list[idx] )
            idx += 1
    return bboxes
def cal_temp(pre=[],hum=[]):

    ret_tp,ret_fn,ret_fp,ret_tn=[0,0,0,0]
    hum = hum.strip('\n')
    if hum[-1]==' ':
        hum=hum[0:len(hum)-1]
    hum = hum.split(' ')
    hum = hum[1:]


    pre = pre.strip('\n')
    pre = pre.split(' ')
    pre = pre[2:]
    #hum = [int(hum[i]) for i in range(len(hum))]
    pre = [float(pre[i]) for i in range(len(pre))]

    # boxes = np.array(hum)
    # boxes = boxes.reshape( -1,4 )
    # format_boxes = boxes
    # format_boxes[:, 0] = boxes[:, 1]
    # format_boxes[:, 1] = boxes[:, 3]
    # format_boxes[:, 2] = format_boxes[:,0] + boxes[:,2]
    # format_boxes[:, 3] = format_boxes[:,1] + boxes[:,0]
    # boxes = format_boxes
    boxes = read_gt_bbox(hum)
    boxes[:, 2] = boxes[:, 2] + boxes[:, 0]
    boxes[:, 3] = boxes[:, 3] + boxes[:, 1]


    pre_boxes = np.array(pre)
    pre_boxes = pre_boxes.reshape( -1, 5)

    for box in pre_boxes :
        #box[2] = box[0]+box[2]
        #box[3] = box[1]+box[3]
        iou = IOU(box,boxes)
        if(np.max(iou) <= 0.65):
            ret_fp = ret_fp + 1
        else :
            ret_tp = ret_tp + 1
    #ret_fn = boxes.shape[0] - ret_tp
    # ret_tn = ?

    return ret_tp,ret_fn,ret_fp,ret_tn
def draw(tpr,fpr):
    import matplotlib.pyplot as plt
    #from sklearn import svm, datasets
    #from sklearn.metrics import roc_curve, auc  ###计算roc和auc
    #from sklearn import cross_validation

    # Import some data to play with
    #iris = datasets.load_iris()
    #X = iris.data
    #y = iris.target

    ##变为2分类
    #X, y = X[y != 2], y[y != 2]

    # Add noisy features to make the problem harder
    # random_state = np.random.RandomState(0)
    # n_samples, n_features = X.shape
    # X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]

    # shuffle and split training and test sets
    # X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=.3, random_state=0)
    #
    # # Learn to predict each class against the other
    # svm = svm.SVC(kernel='linear', probability=True, random_state=random_state)
    #
    # ###通过decision_function()计算得到的y_score的值，用在roc_curve()函数中
    # y_score = svm.fit(X_train, y_train).decision_function(X_test)

    # Compute ROC curve and ROC area for each class
    #fpr, tpr, threshold = roc_curve(y_test, y_score)  ###计算真正率和假正率
    #roc_auc = auc(fpr, tpr)  ###计算auc的值

    plt.figure()
    lw = 2
    plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % -1)#roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()




def show_in_rec(real=0,pic=0,res=0):
    img=cv2.imread(pic)
    if real == 0:
        res = res.split(' ')
        res = res[1:]
        res = [float(res[i]) for i in range(len(res)) ]
        res = np.array(res)
        res = res.reshape(-1,5)
    else :
        res = res.split(' ')
        res = read_gt_bbox(res)
        # res = [int(res[i]) for i in range(len(res))]
        # res = np.array(res)
        # res = res.reshape(-1, 4)
        # t = np.copy(res)
        # res[:,0] = t[:,1]
        # res[:,1] = t[:,3]
        # res[:,3] = t[:,0]
        #
        res[:,2] = res[:,2]+res[:,0]
        res[:,3] = res[:,3]+res[:,1]


    boxes_c = res
    for i in range(boxes_c.shape[0]):
        bbox = boxes_c[i, :4]

        corpbbox = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
        # 画人脸框
        cv2.rectangle(img, (corpbbox[0], corpbbox[1]),
                      (corpbbox[2], corpbbox[3]), (255, 0, 0), 1)
        if real == 0:
            score = boxes_c[i, 4]
            # 判别为人脸的置信度
            cv2.putText(img, '{:.2f}'.format(score),
                        (corpbbox[0], corpbbox[1] - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.imshow('img',img)
    #cv2.imwrite("./rec_show"+str(real)+".jpg",img)
    k = cv2.waitKey(0) & 0xFF
    if k == 27:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    data_dir = "."
    predict_file = "output/wider_face_val_small/wider_face_val_small_test_result.txt"
    human_marked_file = "wider_face_val_small.txt"
    threshold = [0.5, 0.6, 0.65, 0.7, 0.8, 0.9]
    test_draw = False
    test_show_rec = False
    if(test_draw == True):
        tpr=[0.6,0.73,0.82,0.85,0.89,0.91,0.93]
        fpr=[0.2,0.3 ,0.4 ,0.55,0.67,0.79,0.99]
        draw(tpr,fpr)

    if(test_show_rec == True):
        show_in_rec(0,"picture/1_Handshaking_Handshaking_1_664.jpg","6 850 25 896 79 0.999445 450 215 477 246 0.976288 868 30 903 70 0.945503 4 154 46 214 0.858109 449 65 480 101 0.786514 373 130 459 256 0.777324")
        #show_in_rec(1,"picture/1_Handshaking_Handshaking_1_664.jpg","857 651 681 720 740 21 107 118 115 122 42 15 12 12 11 62 22 15 15 17")
        #show_in_rec(1,"picture/1_Handshaking_Handshaking_1_209.jpg","196 616 138 38 94 80 142 176")

    with open(os.path.join(data_dir, predict_file), 'r') as f:
        predict_res = f.readlines()# .split('\\n')
    with open(os.path.join(data_dir, human_marked_file), 'r') as f:
        hum_res = f.readlines() #.split('\\n')
    tp_array = np.array([])
    fp_array = np.array([])


    for thres in threshold:
        tp,fn,fp,tn=[0,0,0,0]
        all_faces_cnt = 0
        pre_faces_cnt = 0
        for i in range(len(predict_res) ):
            temp_tp,temp_fn,temp_fp,temp_tn = cal_temp(predict_res[i],hum_res[i])
            tp = tp + temp_tp
            #fn = fn + temp_fn
            fp = fp + temp_fp
            #tn = tn + temp_tn
            all_faces_cnt = all_faces_cnt + int( (len(hum_res[i])-1)/4)
            pre_faces_cnt = pre_faces_cnt + int( (len(predict_res[i])-2) /5)
        tp_ratio = float(tp) / float(all_faces_cnt)
        fp_ratio = float(fp) / float(pre_faces_cnt) # fp+tn?  tn=?  no tn?
        tp_array = np.append(tp_array,tp_ratio)
        fp_array = np.append(fp_array,fp_ratio)
    draw(tp_array,fp_array)

    print("Draw roc curve done.")

