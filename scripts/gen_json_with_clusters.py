import os
import json
import random
random.seed(1000)
from PIL import Image, ImageDraw
import const
from sklearn.cluster import DBSCAN
import numpy as np 
from tqdm import tqdm

def get_box_feat(box):
    feats = box
    #feats = [np.mean(box[::2]),np.mean(box[1::2])]
    return feats
    
def box_dis(box1, box2):
    def dis(a,b):
        return np.sqrt(np.sum((a-b)*(a-b)))

    pts1 = box1.reshape(-1,2)
    pts2 = box2.reshape(-1,2)
    l = pts1.shape[0]
    dis_ls = []
    for i in range(l):
        for j in range(l):
            dis_ls.append(dis(pts1[i],pts2[j]))
    return np.amin(np.array(dis_ls))

def gen_new_dic(line_dic, added_prop):
    line_clus, line_pos = added_prop
    ret = line_dic
    if len(ret["additional_properties"])>0:
        print(ret)
        raise NotImplementedError
    ret["additional_properties"] = (line_clus, line_pos)
    for cnt in range(len(ret["words"])):
        if len(ret["words"][cnt]["additional_properties"])>0:
            print(ret)
            raise NotImplementedError
        ret["words"][cnt]["additional_properties"] = (line_clus, line_pos, cnt)
    return ret

AZURE_OCR_PATH = const.AZURE_OCR_PATH
OUT_OCR_PATH = const.OUT_OCR_PATH

for _,_,file_ls in os.walk(AZURE_OCR_PATH):
    for fname in tqdm(file_ls):
        iid = fname.split(".json")[0]
        az_ocr_data = json.load(open(os.path.join(AZURE_OCR_PATH,fname)))
        line_ls = az_ocr_data["lines"]

        box_ls = [get_box_feat(line["bounding_box"]) for line in line_ls]
        out_ocr_data = az_ocr_data
        if len(box_ls)>0:
            box_arr = np.array(box_ls)
            if len(box_ls)==1:
                box_arr = box_arr.reshape(1,-1)

            clustering = DBSCAN(eps=100, metric=box_dis, min_samples=1).fit(box_arr)

            new_label_cnt = 0
            label_map_dic = {}
            new_label_ls = []
            for i in clustering.labels_:
                if i==-1:
                    new_label_ls.append((new_label_cnt,0))
                    new_label_cnt+=1
                else:
                    if i not in label_map_dic:
                        label_map_dic[i] = {}
                        label_map_dic[i]["map"] = new_label_cnt
                        label_map_dic[i]["cnt"] = 0
                        new_label_cnt += 1
                    new_label_ls.append((label_map_dic[i]["map"],label_map_dic[i]["cnt"]))
                    label_map_dic[i]["cnt"] += 1

            out_ocr_data["lines"] = []

            zip_ls = list(zip(new_label_ls,line_ls))
            #zip_ls = sorted(zip_ls,key=lambda x:x[0])
            for zipped in zip_ls:
                lb, line_dic = zipped
                out_ocr_data["lines"].append(gen_new_dic(line_dic,lb))
            assert len(new_label_ls)==len(box_ls)
        json.dump(out_ocr_data,open(os.path.join(OUT_OCR_PATH,fname),"w"))



            
