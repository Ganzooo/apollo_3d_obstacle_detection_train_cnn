#!/usr/bin/env python3
# coding: utf-8

import argparse
import os.path

from torch.autograd import Variable
import torch.onnx
import sys
import os
import os.path as osp
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))

from models.BCNN import BCNN
from models.resnet50_unet_activation import UNetWithResnet50Encoder
from collections import OrderedDict


def fix_model_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict['model_state'].items():
        name = k
        if name.startswith('module.'):
            name = name[7:]  # remove 'module.' of dataparallel
        
        new_state_dict[name] = v
    return new_state_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--trained_model', '-tm', type=str,
                        help='trained model',
                        default='/workspace/NETWORK/apollo_3d_obstacle_detection_train_cnn/checkpoints/best_Resnet_UNET/resnet_unet_last_train_52.pth')
    parser.add_argument('--width', type=int,
                        help='feature map width',
                        default=864)
    parser.add_argument('--height', type=int,
                        help='feature map height',
                        default=864)
    parser.add_argument('--channel', type=int,
                        help='feature map channel',
                        default=4)
    parser.add_argument('--model_type', '-mt', type=str,
                        help='model type: unet, resnet_unet',
                        default='resnet_unet')
    args = parser.parse_args()

    ######### Get Model ###########
    if args.model_type == "unet":
        bcnn_model = BCNN(in_channels=args.channel, n_class=5, onnx_export=1)
        #ic('Unet')
    elif args.model_type == "resnet_unet":
        bcnn_model = UNetWithResnet50Encoder(in_channels = args.channel, n_classes=5, onnx_export=1)
        #ic(self.model)
        #ic('ResNet Unet')
    else:
        raise Exception("model selection error!")

    #bcnn_model = BCNN(in_channels=args.channel, n_class=5, onnx_export=1)
    # load it
    state_dict = torch.load(args.trained_model)
    bcnn_model.load_state_dict(fix_model_state_dict(state_dict))
    x = Variable(torch.randn(1, args.channel, args.width, args.height))

    input_names_ = ["data"]
    output_names_ = ["category_score", "instance_pt","confidence_score", "class_score",  "heading_pt", "height_pt"]
    torch.onnx.export(bcnn_model, x, os.path.splitext(
        args.trained_model)[0] + '_new.onnx', verbose=True, input_names=input_names_, output_names=output_names_)
