#!/bin/sh
CUDA_VISIBLE_DEVICES=1 python3 src/train_no_apex_refactor.py --batch_size=22 --max_epoch=150 \
 --data_path=/workspace/dataset/nuScenes/FeatureExtracted/v1.0-mid/ --vis_on=0 \
 --work_dir=./cur/bcnn_mid_db/ --loss_type=BcnnLoss --optimizer adam --scheduler cosine --lr_initial 0.0002