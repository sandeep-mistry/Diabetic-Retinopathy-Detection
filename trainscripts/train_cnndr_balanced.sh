#!/bin/bash
PYTHONPATH=./ \
python3 src/CNNDR.py \
--train \
--epochs 10 \
--lr 0.0001 \
--L2 0.0005 \
--batch_size 32 \
--train_images data/train_512_balanced/ \
--train_csv data/trainLabels_balanced.csv \
--test_images data/train_512_balanced/ \
--test_csv data/trainLabels_balanced.csv \
--save_weights_name weights/weights_balanced_10 \