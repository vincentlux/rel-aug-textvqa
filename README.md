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

# Check prediction results
1. Set path
```
export TVQA_IMG_PATH=<your-directory-with-tvqa-imgs>/train_images
export TVQA_VAL_JSON=<your-directory-with-tvqa-val-json>/TextVQA_0.5.1_val.json

```

2. Run `jupyter notebook` and use `scripts/error_analysis.ipynb` to check predicted results

