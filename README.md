# Brain Tumour MRI Classification
## SE4050 Deep Learning — Group Project

Classifies brain MRI scans into 4 categories: glioma, meningioma, notumor, pituitary.

## Dataset
Download from Kaggle:
https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

Place the extracted folders here (see [data/README.md](data/README.md)):
```
data/Training/
data/Testing/
```

## Setup
```
pip install -r requirements.txt
```
or
```
conda env create -f environment.yml
```

## Run
Open any notebook in `notebooks/` and run all cells top to bottom.

## Random Seeds
All notebooks use seed = 42

## Models
- Custom CNN — Member A
- VGG-16 — Member B
- ResNet-50 — Member C
- EfficientNet-B0 — Member D
