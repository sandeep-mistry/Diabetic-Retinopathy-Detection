#!/usr/bin/env bash
TRAIN_DATA_LINK="https://www.dropbox.com/s/u2wbc1ibql7oxvs/train_512.zip?dl=0"
TRAIN_LABELS_LINK="https://www.dropbox.com/s/bt6ui5w8feb4w72/trainLabels.csv?dl=0"
TEST_DATA_LINK="https://www.dropbox.com/s/ypjwa76uwkk9sb2/test_512.zip?dl=0"
TEST_LABELS_LINK="https://www.dropbox.com/s/fyqbaracxe56mzz/testLabels.csv?dl=0"

TRAIN_DATA_FILE="train_512.zip"
TRAIN_LABELS_FILE="trainLabels.csv"
TEST_DATA_FILE="test_512.zip"
TEST_LABELS_FILE="testLabels.csv"

echo Running setup script
echo Downloading dataset...
wget -nc $TRAIN_DATA_LINK -O $TRAIN_DATA_FILE
wget -nc $TRAIN_LABELS_LINK -O $TRAIN_LABELS_FILE
wget -nc $TEST_DATA_LINK -O $TEST_DATA_FILE
wget -nc $TEST_LABELS_LINK -O $TEST_LABELS_FILE

echo Extracting...
echo Inflating $TRAIN_DATA_FILE
unzip -nq $TRAIN_DATA_FILE
echo Inflating $TEST_DATA_FILE
unzip -nq $TEST_DATA_FILE

echo Done
