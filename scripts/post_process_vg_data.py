import numpy as np
import os
from tqdm import tqdm
import copy

def _expand_annotation(data):
    # expand descriptions (each description becomes one example of self.data)
    print('Start expanding annotation')
    new_data = []
    for d in tqdm(data):
        d.pop('all_normalized_boxes')
        d.pop('index_needed')
        new_d = copy.deepcopy(d)
        for idx, desc in enumerate(d['descriptions']):
            question = desc['txt']
            question_id = desc['question_id']
            new_desc = copy.copy(new_d)
            new_desc['question'] = question
            new_desc['question_id'] = question_id
            new_desc['answers'] = idx
            new_desc['image_id'] = new_desc['feature_path']
            new_desc.pop('descriptions')
            new_data.append(new_desc)
    print(f'original data size: {len(data)} postprocessed data size: {len(new_data)}')
    data = new_data
    return data


if __name__ == '__main__':
    splits = ['train', 'val']
    path = '../data/data/datasets/vg/defaults/annotations/'
    src = os.path.join(path, 'imdb_{}_nms_0.1_rm_dup_sent.npy')
    tgt = os.path.join(path, 'imdb_{}_nms_0.1_rm_dup_sent_v1.npy')

    for split in splits:
        data = np.load(src.format(split), allow_pickle=True)
        v1_data = _expand_annotation(data)
        np.save(tgt.format(split), v1_data, allow_pickle=True)

    print('finished')


