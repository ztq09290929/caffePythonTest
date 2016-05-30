#!/usr/bin/env sh
DIC=/home/ztq/caffe

python $DIC/python/detect.py \
--crop_mode=selective_search \
--pretrained_model=$DIC/models/bvlc_reference_rcnn_ilsvrc13/bvlc_reference_rcnn_ilsvrc13.caffemodel \
--model_def=$DIC/models/bvlc_reference_rcnn_ilsvrc13/deploy.prototxt \
--gpu \
--raw_scale=255 \
$DIC/_temp/det_input.txt \
$DIC/_temp/det_output.h5