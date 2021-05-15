import os
import numpy as np
import torch
from tqdm import tqdm

def norm_box(boxes, im_h, im_w):
    # for m4c
    if not isinstance(boxes, torch.Tensor):
        normalized_boxes = boxes.copy()
    else:
        normalized_boxes = boxes.clone()
    normalized_boxes[:, (0, 2)] /= im_w
    normalized_boxes[:, (1, 3)] /= im_h
    return normalized_boxes


def denorm_box(normed_boxes, im_h, im_w):
    # for converting ocr normalized boxes to original boxes, then do norm_bbox_6
    if not isinstance(normed_boxes, torch.Tensor):
        boxes = normed_boxes.copy()
    else:
        boxes = normed_boxes.clone()
    boxes[:, (0, 2)] *= im_w
    boxes[:, (1, 3)] *= im_h
    return boxes


def norm_box_6(boxes, im_h, im_w):
    '''
    for oscar
    input: 10 * 1d array of len 4 (xmin, ymin, xmax, ymax); img height; img width
    output: np array with shape (num_boxes, 8)
    8: (xmin, ymin, xmax, ymax, xcent, ycent, wbox, hbox) normalized to -1,1
    '''
    # print(f'img width:{im_w} img height:{im_h}')
    # print(f'box 0: {boxes[0]}')
    # print(f'boxes: {boxes}')
    try:
        assert (np.all(boxes[:, 0] <= im_w) and np.all(boxes[:, 2] <= im_w))
        assert (np.all(boxes[:, 1] <= im_h) and np.all(boxes[:, 3] <= im_h))
    except AssertionError as e:
        print(f"data error in {boxes}")

    feats = np.zeros((boxes.shape[0], 6))

    feats[:, 0] = boxes[:, 0] * 2.0 / im_w - 1  # xmin
    feats[:, 1] = boxes[:, 1] * 2.0 / im_h - 1  # ymin
    feats[:, 2] = boxes[:, 2] * 2.0 / im_w - 1  # xmax
    feats[:, 3] = boxes[:, 3] * 2.0 / im_h - 1  # ymax
    # feats[:, 4] = (feats[:, 0] + feats[:, 2]) / 2  # xcenter
    # feats[:, 5] = (feats[:, 1] + feats[:, 3]) / 2  # ycenter
    feats[:, 4] = feats[:, 2] - feats[:, 0]  # box width
    feats[:, 5] = feats[:, 3] - feats[:, 1]  # box height
    return feats


if __name__ == "__main__":
    # when updating obj frcnn lmdb, need to update obj_normalized_boxes
    # field in npy annotation file
    matching_lmdb = "detectron_attrs_max50_v0.lmdb"
    fixed_version = "v0"

    dataset = 'textvqa'

    err_dir = f"data/data/datasets/{dataset}/defaults/annotations"
    err_npys = [
        # "imdb_train_ocr_azure_HQcluster-unsorted",
        # "imdb_val_ocr_azure_HQcluster-unsorted",
        "imdb_test_ocr_azure_HQcluster-unsorted",
        ]
    # dir where obj npz files being saved (from bottom_up_attention.pytorch)
    src_dir = f"/home/vincent/proj/hw/11797/data/{dataset}/test_images_frcnn"

    correct_data_map = {}
    for f in err_npys:
        err_npy = np.load(os.path.join(err_dir, f"{f}.npy"), allow_pickle=True)
        print(err_npy[0].keys())
        for d in tqdm(err_npy):
            if "image_id" not in d:
                print(f"skipped {d}")
                continue
            if dataset == 'textvqa':
                fname = d['image_id']
            elif dataset == 'stvqa':
                fname = d['feature_path'].split('.')[0]

            if fname not in correct_data_map:
                data = np.load(os.path.join(src_dir, f"{fname}.npz"), allow_pickle=True)
                bbox = data["bbox"]
                meta = data["info"].item()
                norm_bbox = norm_box(data["bbox"], meta["image_h"], meta["image_w"])
                # for oscar
                norm_bbox_6 = norm_box_6(data["bbox"], meta["image_h"], meta["image_w"])
                correct_data_map[fname] = {"norm_bbox": norm_bbox, "norm_bbox_6": norm_bbox_6}
            else:
                norm_bbox = correct_data_map[fname]["norm_bbox"]
                norm_bbox_6 = correct_data_map[fname]["norm_bbox_6"]

            # also convert ocr bbox
            if d["ocr_normalized_boxes"].shape[0] != 0:
                ocr_bbox = denorm_box(d["ocr_normalized_boxes"], d["image_height"], d["image_width"])
                norm_ocr_bbox = norm_box_6(ocr_bbox, d["image_height"], d["image_width"])
            else:
                norm_ocr_bbox = np.ones((0,6),np.float32)

            d["obj_normalized_boxes"] = norm_bbox
            d["obj_normalized_boxes_oscar"] = norm_bbox_6
            d["ocr_normalized_boxes_oscar"] = norm_ocr_bbox

        # save to new file
        new_f = os.path.join(err_dir, f"{f}-{fixed_version}.npy")
        np.save(new_f, err_npy)
        print(f"saved {new_f}")
