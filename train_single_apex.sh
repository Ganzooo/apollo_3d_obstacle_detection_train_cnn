#!/bin/sh
CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch --nproc_per_node=1 src/train_apollo_bcnn_ddp_apex.py --batch_size=14 --max_epoch=300 --data_path=/workspace/dataset/nuScenes/FeatureExtracted/v1.0-trainval/ --visualization_on=0 --work_dir=./cur/bcnn_apex/ --loss_type=BcnnLoss
