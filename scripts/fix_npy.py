import numpy as np
import os



if __name__ == '__main__':
    src = 'data/data/datasets/textvqa/defaults/annotations/imdb_{}_ocr_azure.npy'
    tgt = 'data/data/datasets/textvqa/defaults/annotations/imdb_{}_ocr_azure_fixed.npy'
    splits = ['train', 'val']
    for split in splits:

        data = np.load(src.format(split), allow_pickle=True)
        for d in data[1:]:
            if len(d['ocr_normalized_boxes']) == 0:
                d['ocr_normalized_boxes'] = np.ones((0,4),np.float32)

        np.save(tgt.format(split), data, allow_pickle=True)

    print('finished')



