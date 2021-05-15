import lmdb
import pdb
import pickle
import os

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


if __name__ == '__main__':
    # load and see what is the format of open_images/detectron.lmdb
    old_data_name = 'data/data/datasets/stvqa/ocr_en/features/ocr_en_frcn_features.lmdb'
    loader = LMDBLoader(old_data_name)
    for imgid in loader.get_image_ids():
        pdb.set_trace()
        with loader.env.begin(write=False, buffers=True) as txn:
            image_info = pickle.loads(txn.get(imgid))
        print(image_info)
