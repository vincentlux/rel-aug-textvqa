import itertools
import os

from tqdm import tqdm
import lmdb
import numpy as np
import pickle

def dict_to_lmdb(out_path, dict_to_save):
    env = lmdb.open(out_path, map_size=1.5e+10, writemap=True)
    with env.begin(write=True) as txn:
        keys = []
        for i, (k, v) in enumerate(dict_to_save.items()):
            assert txn.put(k.encode(), pickle.dumps(v)), 'failed to add {} into lmdb'.format(k)
            keys.append(k.encode())

        txn.put('keys'.encode(), pickle.dumps(keys))
        print("added {} records to the database".format(len(keys)))


if __name__ == '__main__':
    # TODO: consider regenerate lmdb to include train+val+test?
    # load npy file to get filename
    file_path1 = "data/data/datasets/textvqa/"
    # file_path2 = "/bos/tmp6/zhenfan/VQA/TextVQA_orig/"
    train_npy_path = f"{file_path1}defaults/annotations/imdb_train_ocr_azure-clus-unsorted-v0.npy"
    val_npy_path = f"{file_path1}defaults/annotations/imdb_val_ocr_azure-clus-unsorted-v0.npy"
    test_npy_path = f"{file_path1}defaults/annotations/imdb_test_ocr_azure-v0.npy"

    base_feat_path = f"{file_path1}ocr_azure/features/ocr_azure_frcn_features_all/"
    save_lmdb_name = f"{file_path1}ocr_azure/features/ocr_azure_frcn_features_all.lmdb"

    train_info = np.load(train_npy_path, allow_pickle=True)[1:]
    val_info = np.load(val_npy_path, allow_pickle=True)[1:]
    test_info = np.load(test_npy_path, allow_pickle=True)[1:]
    # get all filename needed and extract filename.npy features to lmdb
    feature_paths = list(set([i['feature_path'] for i in itertools.chain(train_info, val_info, test_info)]))
    # record test paths
    test_paths = set([i['feature_path'] for i in test_info])
    print('Num image feats: {}'.format(len(feature_paths)))
    # load all feat to a dict
    # imgid: train/imgid
    imgid_feat_map = {}

    for p in tqdm(feature_paths):
        filename = os.path.join(base_feat_path, p)
        if not os.path.isfile(filename):
            import pdb; pdb.set_trace()
        data = np.load(filename)
        split = 'train' if p not in test_paths else 'test'
        imgid = '{}/{}'.format(split, p.split('.')[0])
        if split == 'test':
            print(imgid)
        feat = {'features': data, 'source': 'azure'}
        imgid_feat_map[imgid] = feat

    dict_to_lmdb(save_lmdb_name, imgid_feat_map)

    print('finished')










