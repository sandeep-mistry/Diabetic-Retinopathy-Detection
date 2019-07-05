#!/bin/bash
PYTHONPATH=./ \
python3 src/evaluation/EvaluateModel.py \
--load_weights_name weights/weights_balanced_alex_10 \
--alexnet \
--test_images data/test_512/ \
--test_csv data/testLabels.csv \
--batch_size 32 \
