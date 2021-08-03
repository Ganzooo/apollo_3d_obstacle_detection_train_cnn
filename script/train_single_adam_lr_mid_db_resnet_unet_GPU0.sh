#!/bin/sh
CUDA_VISIBLE_DEVICES=0 python3 src/train_no_apex_refactor_apollo_baidu.py --batch_size=5 --max_epoch=150 \
 --data_path=/workspace/dataset/nuScenes/FeatureExtracted/v1.0-mid/ --vis_on=0 \
 --work_dir=./cur/bcnn_mid_db_resnet_unet_000_2/ --loss_type=BcnnLoss --optimizer adam --scheduler cosine --lr_initial 0.0002 \
 --model_type resnet_unet