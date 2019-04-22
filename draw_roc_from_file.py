import  os
import numpy as np
from preprocess.utils import IOU

def get_TP():
    TP = 0
    return TP

def get_FP():
    FP = 0
    return FP

def cal_temp(pre=[],hum=[]):

    ret_tp,ret_fn,ret_fp,ret_tn=[0,0,0,0]
    boxes = np.array(hum[1:])
    boxes = boxes.reshape( -1,4 )

    pre_boxes = np.array(pre[2:] )
    pre_boxes = pre_boxes.reshape( -1, 5) # also

    for box in pre_boxes :
        iou = IOU(box,boxes)
        if(np.max(iou) <= 0.65):
            ret_fp = ret_fp + 1
        else :
            ret_tp = ret_tp + 1
    ret_fn = boxes.shape[0] - ret_tp

    # ret_tn = ?

    return ret_tp,ret_fn,ret_fp,ret_tn


if __name__ == "__main__":
    data_dir = "."
    predict_file = "output/wider_face_val_small/wider_face_val_small_test_result.txt"
    human_marked_file = "wider_face_val.txt"

    with open(os.path.join(data_dir, predict_file), 'r') as f:
        predict_res = f.readlines()
    with open(os.path.join(data_dir, human_marked_file), 'r') as f:
        hum_res = f.readlines()
    tp,fn,fp,tn=[0,0,0,0]
    all_faces_cnt = 0
    for i in range(len(predict_res) ):
        temp_tp,temp_fn,temp_fp,temp_tn = cal_temp(predict_res[i],hum_res[i])
        tp = tp + temp_tp
        fn = fn + temp_fn
        fp = fp + temp_fp
        tn = tn + temp_tn
        all_faces_cnt = all_faces_cnt + int((len(hum_res[i])-1)/4)

    tp_ratio = float(tp) / float(all_faces_cnt)
    fp_ratio = float(fp) / float(fp+tn) # fp+tn?  tn=?  no tn?
    pass

