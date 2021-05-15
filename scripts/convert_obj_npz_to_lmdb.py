import lmdb
import numpy as np
import os
import pdb
import pickle
from tqdm import tqdm
from scripts.convert_npy_to_lmdb import dict_to_lmdb



def load_vocab_file(filename):
    objects = []
    with open(filename) as f:
        for i in f.readlines():
            objects.append(i.strip())
    id_objects_map = {i: v for i, v in enumerate(objects)}
    return id_objects_map

def get_imgids(npy_path):
    info = np.load(npy_path, allow_pickle=True)[1:]
    imgids = set([i['feature_path'].split('.')[0] for i in info])
    return imgids


if __name__ == '__main__':

    # ['feature_path', 'features', 'image_height', 'image_width', 'num_boxes', 'objects', 'cls_prob', 'bbox']
    dataset = 'stvqa' # 'textvqa'
    if dataset == 'stvqa':
        mode = None # stvqa does not need append mode in the begining
        npz_folder = '/home/vincent/proj/hw/11797/data/stvqa/train_images_frcnn/'
        tgt_name = 'data/data/datasets/stvqa/defaults/features/detectron_attrs_max50_v0.lmdb'
        npz_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(npz_folder) for f in fn]
        # silly way of removing root path
        npz_files = [i.replace(npz_folder, '') for i in npz_files]
        # need to judge if it is train or test_task3
        test_npy_path = f"data/data/datasets/stvqa/defaults/annotations/azure_st_subtest.npy"
        test_imgids = get_imgids(test_npy_path)

    elif dataset == 'textvqa':
        mode = 'test' # 'train'
        npz_folder = '/home/vincent/proj/hw/11797/data/textvqa/test_images_frcnn'
        tgt_name = 'data/data/datasets/textvqa/defaults/features/open_images/detectron_attrs_max50_v0_test.lmdb'
        npz_files = os.listdir(npz_folder)

    vg_objs = load_vocab_file('scripts/objects_vocab.txt')
    vg_attr = load_vocab_file('scripts/attributes_vocab.txt')

    key_info_map = {}
    for npz in tqdm(npz_files):
        if npz.endswith('npz'):
            filename = os.path.join(npz_folder, npz)
            data = np.load(filename, allow_pickle=True)
            meta = data['info'].item()

            features = data['x']
            bbox = data['bbox']
            # for stvqa, we need to judge whether one imgid belong to train or test_task3
            if dataset == 'stvqa':
                mode = 'test_task3' if meta['image_id'] in test_imgids else 'train'
            key = '{}/{}'.format(mode, meta['image_id'])
            feature_path = '{}/{}'.format(mode, meta['image_id'])

            image_height = meta['image_h']
            image_width = meta['image_w']
            num_boxes = meta['num_boxes']
            objects = meta['objects_id']
            attrs = meta['attrs_id']
            object_tokens = [vg_objs[i] for i in objects]
            attr_tokens = [vg_attr[i] for i in attrs]

            key_info_map[key] = {
                'feature_path': feature_path,
                'features': features,
                'image_height': image_height,
                'image_width': image_width,
                'num_boxes': num_boxes,
                'objects': objects,
                'attrs': attrs,
                'object_tokens': object_tokens,
                'attr_tokens': attr_tokens
            }

    dict_to_lmdb(tgt_name, key_info_map)












