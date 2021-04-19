import numpy as np
import json
import os
from tqdm import trange

set_str = "val"
sort_str = "unsorted"
in_dir = f"/mnt/d/Downloads/VQA/imdb_{set_str}_ocr_en.npy"
out_dir = f"/mnt/d/Downloads/VQA/imdb_{set_str}_ocr_azure-clus-{sort_str}.npy"
json_dir = f"/mnt/d/Downloads/train_images_AZOCR_cluster-{sort_str}/"
if __name__ == "__main__":
    ori_data = np.load(in_dir, allow_pickle=True)
    new_data = ori_data
    for i in trange(1, new_data.shape[0], ncols=100):
        try:
            with open(f"{json_dir}/{new_data[i]['image_id']}.json", 'r') as fin:
                azr_data = json.load(fin)
        except:
            print(f"can't find file for {new_data[i]['image_id']}")
            raise NotImplementedError
        image_width = azr_data["width"]
        image_height = azr_data["height"]
        ocr_tokens = []
        ocr_info = []
        ocr_normalized_boxes = []
        for line in azr_data["lines"]:
            for word in line["words"]:
                ocr_tokens.append(word['text'])
                ocr_normalized_boxes.append([
                    min(word['bounding_box'][0], word['bounding_box'][2], word['bounding_box'][4], word['bounding_box'][6]) / image_width,
                    min(word['bounding_box'][1], word['bounding_box'][3], word['bounding_box'][5], word['bounding_box'][7]) / image_height,
                    max(word['bounding_box'][0], word['bounding_box'][2], word['bounding_box'][4], word['bounding_box'][6]) / image_width,
                    max(word['bounding_box'][1], word['bounding_box'][3], word['bounding_box'][5], word['bounding_box'][7]) / image_height
                ])
                ocr_info.append({
                    'word': word['text'],
                    'additional_properties': word["additional_properties"],
                    'bounding_box': {
                        'topLeftX': word['bounding_box'][0] / image_width,
                        'topLeftY': word['bounding_box'][1] / image_height,
                        'width': np.sqrt(
                            ((word['bounding_box'][2] - word['bounding_box'][0]) / image_width)
                          * ((word['bounding_box'][2] - word['bounding_box'][0]) / image_width)
                          + ((word['bounding_box'][3] - word['bounding_box'][1]) / image_height)
                          * ((word['bounding_box'][3] - word['bounding_box'][1]) / image_height)
                        ),
                        'height': np.sqrt(
                            ((word['bounding_box'][4] - word['bounding_box'][2]) / image_width)
                          * ((word['bounding_box'][4] - word['bounding_box'][2]) / image_width)
                          + ((word['bounding_box'][5] - word['bounding_box'][3]) / image_height)
                          * ((word['bounding_box'][5] - word['bounding_box'][3]) / image_height)
                        ),
                        'rotation': 0,
                        'roll': 0,
                        'pitch': 0,
                        'yaw': np.pi / 2 * np.sign(word['bounding_box'][3] - word['bounding_box'][1]) if word['bounding_box'][2] - word['bounding_box'][0] == 0
                            else np.arctan(
                                (word['bounding_box'][3] - word['bounding_box'][1]) / (word['bounding_box'][2] - word['bounding_box'][0])
                            )
                    }
                })
        new_data[i]['ocr_tokens'] = ocr_tokens
        new_data[i]['ocr_info'] = ocr_info
        norm_box = np.array(ocr_normalized_boxes, dtype=np.float32)
        if len(norm_box) == 0:
            norm_box = np.ones((0,4),np.float32)
        new_data[i]['ocr_normalized_boxes'] = norm_box
    np.save(out_dir, new_data)
