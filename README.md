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
