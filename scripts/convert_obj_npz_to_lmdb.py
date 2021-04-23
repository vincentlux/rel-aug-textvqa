import lmdb
import numpy as np
import os
import pdb
import pickle
from tqdm import tqdm
from scripts.convert_npy_to_lmdb import dict_to_lmdb


class LMDBLoader:
    def __init__(self, db_path):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        self.env = lmdb.open(
            self.db_path,
            subdir=os.path.isdir(self.db_path),
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with self.env.begin(write=False, buffers=True) as txn:
            self.image_ids = pickle.loads(txn.get(b"keys"))
            self.image_id_indices = {
                self.image_ids[i]: i for i in range(0, len(self.image_ids))
            }

    def get_image_ids(self):
        return self.image_ids



def load_vocab_file(filename):
    objects = []
    with open(filename) as f:
        for i in f.readlines():
            objects.append(i.strip())
    id_objects_map = {i: v for i, v in enumerate(objects)}
    return id_objects_map

if __name__ == '__main__':
    # load and see what is the format of open_images/detectron.lmdb
    # old_data_name = 'data/data/datasets/textvqa/defaults/features/open_images/detectron.lmdb'
    # loader = LMDBLoader(old_data_name)
    # for imgid in loader.get_image_ids():
    #     pdb.set_trace()
    #     with loader.env.begin(write=False, buffers=True) as txn:
    #         image_info = pickle.loads(txn.get(imgid))
    #     print(image_info)

    # ['feature_path', 'features', 'image_height', 'image_width', 'num_boxes', 'objects', 'cls_prob', 'bbox']
    npz_folder = '/home/vincent/proj/hw/11797/data/textvqa/train_images_frcnn'
    tgt_name = 'data/data/datasets/textvqa/defaults/features/open_images/detectron_attrs_max50_v0.lmdb'

    vg_objs = load_vocab_file('scripts/objects_vocab.txt')
    vg_attr = load_vocab_file('scripts/attributes_vocab.txt')

    key_info_map = {}
    for npz in tqdm(os.listdir(npz_folder)):
        if npz.endswith('npz'):
            filename = os.path.join(npz_folder, npz)
            data = np.load(filename, allow_pickle=True)
            meta = data['info'].item()

            features = data['x']
            bbox = data['bbox']
            key = 'train/{}'.format(meta['image_id'])
            feature_path = 'train/{}'.format(meta['image_id'])
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












