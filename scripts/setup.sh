# !/bin/bash

ZIP_FILE=data.zip
DATA_DIR=data
ENV=AML

echo Running setup script

if [ ! -f $ZIP_FILE ]; then
    echo Downloading dataset...
    wget "https://www.dropbox.com/s/tsx9zdvcw3wnxzp/All_Data.zip?dl=0" -O $ZIP_FILE
else
    echo Dataset already downloaded
fi

echo Extracting...
if [ -d "data" ]; then
    read -p "Data folder found, overwrite? (y/n)" -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo 'Removing existing'
        rm -rf data
    else
        echo 'Aborting'
        exit
    fi
fi

echo Inflating zip file
mkdir $DATA_DIR
unzip -q $ZIP_FILE -d $DATA_DIR

echo Grabbing training data
mv 'data/All_Data/Train_Processed' 'train_sub'
echo Grabbing test data
mv 'data/All_Data/Test_Processed' 'test_sub'

echo Creating environment
conda env remove --name $ENV
conda create --yes --name $ENV

echo Cleaning up
rm -rf $DATA_DIR
rm $ZIP_FILE

echo Setup complete
