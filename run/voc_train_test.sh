#!/usr/bin/env bash

python /app/train.py \
	--dataset /app/data/voc_train.tfrecord \
	--val_dataset ./data/voc_val.tfrecord \
	--classes /app/data/voc2012.names \
	--num_classes 20 \
	--mode fit --transfer darknet \
	--batch_size 16 \
	--epochs 3 \
	--weights /app/checkpoints/yolov3.tf \
	--weights_num_classes 80 