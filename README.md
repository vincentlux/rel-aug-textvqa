## TextVQA Project 


# Installation

```
pip install -e .

pip install -r requirements.txt
```

---

# Run m4c baseline

```
train:
python run.py config=configs/m4c/configs/textvqa/joint_with_stvqa.yaml datasets=textvqa model=m4c run_type=train


val:
python run.py config=configs/m4c/configs/textvqa/joint_with_stvqa.yaml datasets=textvqa model=m4c run_type=val  env.save_dir=./snap checkpoint.resume=True checkpoint.resume_best=True evaluation.predict=True
```


---

# Run m4c+azure ocr (no stvqa)

1. download dataset

a. put azure ocr [train npy](https://vincent-research.s3.amazonaws.com/2021-rel-aug-textvqa/data/data/datasets/textvqa/defaults/annotations/imdb_train_ocr_azure.npy) and [val npy](https://vincent-research.s3.amazonaws.com/2021-rel-aug-textvqa/data/data/datasets/textvqa/defaults/annotations/imdb_val_ocr_azure.npy) into 

```
data/data/datasets/textvqa/defaults/annotations/
```

b. put azure ocr [frcnn feature](https://vincent-research.s3.amazonaws.com/2021-rel-aug-textvqa/data/data/datasets/textvqa/ocr_azure/features/ocr_azure_frcn_features.lmdb.tar.gz) into 

```
data/data/datasets/textvqa/ocr_azure/features/
```

then `tar -xzvf ocr_azure_frcn_features.lmdb.tar.gz` 


2. start training

```
python run.py config=configs/m4c/configs/textvqa/defaults_azure.yaml datasets=textvqa model=m4c run_type=train

```

# Track experiment results and params
Run mlflow under current directory:
```
mlflow server --host 0.0.0.0:5000
```


---

# Check prediction results
1. Set path
```
export TVQA_IMG_PATH=<your-directory-with-tvqa-imgs>/train_images
export TVQA_VAL_JSON=<your-directory-with-tvqa-val-json>/TextVQA_0.5.1_val.json

```

2. Run `jupyter notebook` and use `scripts/error_analysis.ipynb` to check predicted results

