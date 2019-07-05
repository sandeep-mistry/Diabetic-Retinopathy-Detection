#!/usr/bin/env bash
python3 CNNDR.py \
--vgg \
 --train \
  --pretrained \
 --epochs 10 \
 --lr 0.001 \
 --L2 0.0005 \
 --batch_size 32 \
 --test_images data/test_512/ \
 --train_images data/train_512_balanced/ \
 --test_csv data/testLabels.csv \
 --train_csv data/trainLabels_balanced.csv \
 --cuda 0 \
 --save_weights_name 0206TransferVGG \
 --balanced \
 --transform 1
