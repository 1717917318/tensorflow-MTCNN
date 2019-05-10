import  os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from preprocess.utils import IOU

def read_gt_bbox(raw_list):
    bbox_num = len(raw_list) //4
    idx = 0
    bboxes = np.zeros( (bbox_num,4), dtype=int)
    for i in range(bbox_num):
        for j in range(4):
            bboxes[i][j] = int (raw_list[idx] )
            idx += 1
    return bboxes

def refine_line(pre,hum):  # 将每行结果去掉换行并转换为float类型的list
    pre = pre.strip('\n')
    pre = pre.split(' ')
    pre = pre[2:]
    pre = [float(pre[i]) for i in range(len(pre))]

    hum = hum.strip('\n')
    if hum[-1] == ' ':
        hum = hum[0:len(hum) - 1]
    hum = hum.split(' ')
    hum = hum[1:]

    return pre,hum

def cal_temp(pre=[],hum=[],w_f=False):

    ret_tp,ret_fn,ret_fp,ret_tn=[0,0,0,0]

    pre, hum =  refine_line(pre,hum)

    boxes = read_gt_bbox(hum)
    if w_f:
        boxes[:, 2] = boxes[:, 2] + boxes[:, 0]
        boxes[:, 3] = boxes[:, 3] + boxes[:, 1]


    pre_boxes = np.array(pre)
    pre_boxes = pre_boxes.reshape( -1, 5)

    for box in pre_boxes :
        iou = IOU(box,boxes)
        if(np.max(iou) <= 0.5): #一般认为 iou大于0.5就算检测出来了
            ret_fp = ret_fp + 1
        else :
            ret_tp = ret_tp + 1
    return ret_tp,ret_fn,ret_fp,ret_tn

def save_img(tmethod, pre,hum):
    hum_copy = hum
    hum_copy = hum_copy.split(' ')
    imgpath = hum_copy[0]
    pics_path = '../DATA/FDDB/originalPics'

    img = cv2.imread(pics_path+'/'+imgpath+'.jpg' )

    pre, hum = refine_line(pre, hum)
    pre = [int(pre[i]) for i in range(len(pre))]
    boxes = read_gt_bbox(hum)

    if w_f:
        boxes[:, 2] = boxes[:, 2] + boxes[:, 0]
        boxes[:, 3] = boxes[:, 3] + boxes[:, 1]

    pre_boxes = np.array(pre)
    pre_boxes = pre_boxes.reshape( -1, 5)
    for corpbbox in pre_boxes:
        cv2.rectangle(img, (corpbbox[0], corpbbox[1]),
                      (corpbbox[2], corpbbox[3]), (255, 0, 0), 1)

    for corpbbox in boxes:
        cv2.rectangle(img, (corpbbox[0], corpbbox[1]),
                      (corpbbox[2], corpbbox[3]), (0, 0, 255), 1)

    output_path =  'output/'+tmethod
    if (not os.path.exists(output_path) ):
        os.mkdir(output_path)

    imgpath = imgpath.split('/')
    cv2.imwrite(output_path+'/'+imgpath[-1]+'.jpg',img)

def draw(tpr,fpr):
    import matplotlib.pyplot as plt
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

def get_tp_fp(method,data_dir,predict_file,human_marked_file,w_f):

    threshold = [0.5, 0.6, 0.65, 0.7, 0.8, 0.9]

    with open(os.path.join(data_dir, predict_file), 'r') as f:
        predict_res = f.readlines()# .split('\\n')
    with open(os.path.join(data_dir, human_marked_file), 'r') as f:
        hum_res = f.readlines() #.split('\\n')
    tp_array = np.array([])
    fp_array = np.array([])

    acc_seq  = np.array([])
    fp_seq = np.array([])


    for thres in threshold:
        tp,fn,fp,tn=[0,0,0,0]
        all_faces_cnt = 0
        pre_faces_cnt = 0

        all_acc_seq = 0
        all_fp_seq = 0

        for i in range(len(predict_res) ):

            if i>180 :
                break

            temp_tp,temp_fn,temp_fp,temp_tn = cal_temp(predict_res[i],hum_res[i],w_f)

            if write_img:
                save_img(method,predict_res[i],hum_res[i])

            tp = tp + temp_tp
            #fn = fn + temp_fn
            fp = fp + temp_fp
            #tn = tn + temp_tn
            spre= predict_res[i][0:-1]
            spre= spre.split(' ')
            shum = hum_res[i][0:-1]
            shum = shum.split(' ')
            temp_pre_faces_cnt = (len(spre)-2 ) /5
            temp_all_faces_cnt = int( (len(shum)-1)/4)

            temp_acc_seq = temp_tp/temp_all_faces_cnt
            temp_fp_seq =  temp_fp

            all_acc_seq = (i*all_acc_seq+temp_acc_seq)/(i+1)
            all_fp_seq = all_fp_seq + temp_fp_seq
            acc_seq = np.append(acc_seq,all_acc_seq)
            fp_seq = np.append(fp_seq,all_fp_seq)

            all_faces_cnt = all_faces_cnt + temp_all_faces_cnt
            pre_faces_cnt = pre_faces_cnt + int( temp_pre_faces_cnt)
            if i%100 == 0:
                print(str(i)+"has done.")


        tp_ratio = float(tp) / float(all_faces_cnt)
        fp_ratio = float(fp) / float(pre_faces_cnt) # fp+tn?  tn=?  no tn?
        tp_array = np.append(tp_array,tp_ratio)
        fp_array = np.append(fp_array,fp_ratio)


        break

    return fp_seq, acc_seq

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
    k = cv2.waitKey(0) & 0xFF
    if k == 27:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    data_dir = "."
    w_f =  False # dataset is not wider_face_val
    write_img = False
    human_marked_file = "wider_face_val.txt"
    if not w_f:
        human_marked_file = "test/fddb_val_min_r_small.txt"
    # predict_file = "output/wider_face_val_small/wider_face_val_small_test_result.txt"
    # human_marked_file = "wider_face_val_small.txt"
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


    predict_file = "output/wider_face_valmtcnn/wider_face_valmtcnn_test_result.txt"
    method = 'mtcnn'
    if not w_f:
        # predict_file = 'output/fddb_valmtcnn/fddb_valmtcnn_test_result_thresh3_0.7.txt'
        #predict_file = 'output/fddb_val_min_rmtcnn/fddb_val_min_rmtcnn_test_result_thresh3_0.35.txt'
        #predict_file =  'output/fddb_val_min_rmtcnn/fddb_val_min_rmtcnn_test_result.txt'
        predict_file = 'output/fddb_val_min_r_smallmtcnn/fddb_val_min_r_smallmtcnn_test_result.txt'
    fp_seq, acc_seq = get_tp_fp(method,data_dir,predict_file, human_marked_file,w_f)
    #draw(tp_array,fp_array)
    #draw_acc_fp(acc_seq,fp_seq)
    plt.figure()
    lw = 2
    plt.figure(figsize=(10, 10))
    plt.plot(fp_seq, acc_seq, color='darkorange',
             lw=lw, label='mtcnn_acc_fp curve (area = %0.2f)' % -1)  # acc_fp)  ###fp为横坐标，acc为纵坐标做曲线



    predict_file = "output/wider_face_valhaar/wider_face_valhaar_test_result.txt"
    method = 'haar'
    if not w_f:
        predict_file = 'output/fddb_valhaar/fddb_valhaar_test_result.txt'
    fp_seq, acc_seq = get_tp_fp(method,data_dir, predict_file, human_marked_file,w_f)
    plt.plot(fp_seq, acc_seq, color='blue',
             lw=lw, label='haar_acc_fp curve (area = %0.2f)' % -1)  # acc_fp)  ###fp为横坐标，acc为纵坐标做曲线

    predict_file = "output/wider_face_valdnn/wider_face_valdnn_test_result.txt"
    method = 'dnn'
    if not w_f:
        predict_file = 'output/fddb_valdnn/fddb_valdnn_test_result.txt'
    fp_seq, acc_seq = get_tp_fp(method,data_dir, predict_file, human_marked_file,w_f)
    plt.plot(fp_seq, acc_seq, color='red',
             lw=lw, label='dnn_acc_fp curve (area = %0.2f)' % -1)  # acc_fp)  ###fp为横坐标，acc为纵坐标做曲线

    predict_file = "output/wider_face_valhog/wider_face_valhog_test_result.txt"
    method = 'hog'
    if not w_f:
        predict_file = 'output/fddb_valhog/fddb_valhog_test_result.txt'
    fp_seq, acc_seq = get_tp_fp(method,data_dir, predict_file, human_marked_file,w_f)
    plt.plot(fp_seq, acc_seq, color='black',
             lw=lw, label='hog_acc_fp curve (area = %0.2f)' % -1)  # acc_fp)  ###fp为横坐标，acc为纵坐标做曲线

    predict_file = "output/wider_face_valmmod/wider_face_valmmod_test_result.txt"
    method = 'mmod'
    if not w_f:
        predict_file = 'output/fddb_valmmod/fddb_valmmod_test_result.txt'
    fp_seq, acc_seq = get_tp_fp(method, data_dir, predict_file, human_marked_file,w_f)
    plt.plot(fp_seq, acc_seq, color='green',
             lw=lw, label='mmod_acc_fp curve (area = %0.2f)' % -1)  # acc_fp)  ###fp为横坐标，acc为纵坐标做曲线

    plt.xlabel('False Positive numbers')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    print("Draw tp_fp curve done.")

