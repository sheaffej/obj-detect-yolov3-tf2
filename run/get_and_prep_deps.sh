#!/usr/bin/env bash

cd /app

# Download and convert Pretrained Darknet Weight
wget https://pjreddie.com/media/files/yolov3.weights -O data/yolov3.weights
python convert.py

# Download VOC 2009 image dataset
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2009/VOCtrainval_11-May-2009.tar -O data/voc2009_raw.tar
mkdir -p data/voc2009_raw
tar -xvf data/voc2009_raw.tar -C ./data/voc2009_raw
# rm -f data/voc2009_raw.tar

# Create tfrecord for training set
python tools/voc2012.py \
  --data_dir 'data/voc2009_raw/VOCdevkit/VOC2009' \
  --split train \
  --output_file data/voc_train.tfrecord

# Create tfrecord for validation set
python tools/voc2012.py \
  --data_dir 'data/voc2009_raw/VOCdevkit/VOC2009' \
  --split val \
  --output_file data/voc_val.tfrecord

