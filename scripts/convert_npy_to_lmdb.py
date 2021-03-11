import itertools
import os

from tqdm import tqdm
import lmdb
import numpy as np
import pickle

def dict_to_lmdb(out_path, dict_to_save):
    env = lmdb.open(out_path, map_size=1099511627776, writemap=True)
    with env.begin(write=True) as txn:
        keys = []
        for i, (k, v) in enumerate(dict_to_save.items()):
            assert txn.put(k.encode(), pickle.dumps(v)), 'failed to add {} into lmdb'.format(k)
            keys.append(k.encode())

        txn.put('keys'.encode(), pickle.dumps(keys))
        print("added {} records to the database".format(len(keys)))


if __name__ == '__main__':
    # load npy file to get filename
    train_npy_path = 'data/data/datasets/textvqa/defaults/annotations/imdb_train_ocr_azure.npy'
    val_npy_path = 'data/data/datasets/textvqa/defaults/annotations/imdb_val_ocr_azure.npy'
    base_feat_path = 'data/data/datasets/textvqa/ocr_azure/features/ocr_azure_frcn_features/'
    save_lmdb_name = 'data/data/datasets/textvqa/ocr_azure/features/ocr_azure_frcn_features.lmdb'
    train_info = np.load(train_npy_path, allow_pickle=True)[1:]
    val_info = np.load(val_npy_path, allow_pickle=True)[1:]
    # get all filename needed and extract filename.npy features to lmdb
    feature_paths = list(set([i['feature_path'] for i in itertools.chain(train_info, val_info)]))
    print('Num image feats: {}'.format(len(feature_paths)))
    # load all feat to a dict
    # imgid: train/imgid
    imgid_feat_map = {}

    for p in tqdm(feature_paths):
        filename = os.path.join(base_feat_path, p)
        if not os.path.isfile(filename):
            import pdb; pdb.set_trace()
        data = np.load(filename)
        imgid = 'train/{}'.format(p.split('.')[0])
        feat = {'features': data, 'source': 'azure'}
        imgid_feat_map[imgid] = feat

    dict_to_lmdb(save_lmdb_name, imgid_feat_map)

    print('finished')










