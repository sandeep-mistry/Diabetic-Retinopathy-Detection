# Diabetic Retinopathy Detection Kaggle Competition

# Introduction

We are given a large set of high-resolution retina images. 
A left and right field is provided for every subject. 
A clinician has rated the presence of diabetic retinopathy in each training image on a scale from 0 to 4, 
where 0 indicates no diabetic retinopathy detected and 4 indicates proliferative diabetic retinopathy detected. 
Our task is to create a system able to assign a score to new test images based on this scale.

Link to competition: https://www.kaggle.com/c/diabetic-retinopathy-detection


# Requirements
```
pip install -r requirements.txt
```
# Usage

Clone repository.

Download weights.tar.gz file from [here](https://drive.google.com/file/d/1nE1eZL-TvlJSalmm1Z_NXEw-qoicEEfj/view?usp=sharing) and extract to 'weights' folder in repository.

Set train and test directories in chosen classification model script.
```
python [model].py e.g. RandomForest.py
```
