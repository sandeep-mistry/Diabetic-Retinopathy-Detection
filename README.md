Diabetic Retinopathy Detection

We are given a large set of high-resolution retina images. 
A left and right field is provided for every subject. 
A clinician has rated the presence of diabetic retinopathy in each training image on a scale from 0 to 4, 
where 0 indicates no diabetic retinopathy detected and 4 indicates proliferative diabetic retinopathy detected. 
Our task is to create a system able to assign a score to new test images based on this scale.

Our Aim is to use multiple techniques to classify our data: 
•	SVM (one vs all)
•	Logistic Regressions 
•	Random Forest Technique
•	Deep Learning Neural Networks
•	Noise Reduction Techniques

Down-sampling the images (due to high resolution)

Challenge?
•	Robust to inversion (Different types of cameras used to obtain images)
•	Robust to noise (blurring changes in sharpness)
•	80GB, subsample, validation-test-training

•	Approach, systematically
•	Transfer Learning, use pre-trained models
•	Analyse Techniques
•	What works, what doesn't
•	Re-use
•	Deep Learning as feature extractor with SVM classifier (or other) works well.
•	Features extracted? What are the features/ network identifying?
