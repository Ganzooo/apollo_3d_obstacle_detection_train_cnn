#!/usr/bin/env python3
# coding: utf-8

import argparse
import os.path as osp
import sys
import os
from pathlib import Path
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import tqdm
from datetime import datetime
import random
from collections import OrderedDict

sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))

from dataset.NuscData import load_dataset
from models.BCNN import BCNN
from loss import get_loss_function

from utils.visualize_utils import get_arrow_image, get_class_image, get_category_or_confidence_image, get_input_feature_image  # noqa
from utils.utils import get_logger
from icecream import ic
import cv2
import options
import visdom

#import apex
#from apex.parallel import DistributedDataParallel as DDP
#from apex.fp16_utils import *
#from apex import amp, optimizers
from warmup_scheduler import GradualWarmupScheduler
from torch.optim.lr_scheduler import StepLR

######### Set Seeds ###########
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight)

class Trainer(object):
    def __init__(self, args):
        self.data_path=args.data_path,
        self.batch_size=args.batch_size,
        self.max_epoch=args.max_epoch,
        self.pretrained_weight=args.pretrained_model,
        self.width=args.width,
        self.height=args.height,
        self.resume_train = args.resume,
        self.work_dir = args.work_dir,
        self.train_dataloader, self.val_dataloader = load_dataset(args.data_path, args.batch_size, args.distributed)
        self.max_epoch = args.max_epoch
        self.time_now = datetime.now().strftime('%Y%m%d_%H%M')
        self.width = args.width
        self.height = args.height
        self.args = args
        
        self.best_loss = 1e10
        self.vis_on = args.vis_on
        if self.vis_on:
            self.vis = visdom.Visdom()
            self.vis_interval = 1
        
        self.in_channels = 4
        self.non_empty_channle = 3

        self.save_model_interval = 1
        self.loss_print_interval = 1
        self.start_epo = 0

        self.grid_range = 90.

        self.logdir = osp.join("./", args.work_dir)
        self.logger = get_logger(self.logdir)
        self.logger.info("Let the train begin...")

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        ######### Get Model ###########
        if args.model_type == "unet":
            self.model = BCNN(in_channels=self.in_channels, n_class=5).to(self.device)
            ic('Unet')
        else:
            raise Exception("model selection error!")
        self.model.apply(weights_init)
        ic('Total parameter of model', sum(p.numel() for p in self.model.parameters()))

        ######### Initialize Optimizer ###########
        if args.optimizer == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr_initial, betas=(0.9, 0.999),eps=1e-8, weight_decay=args.weight_decay)
        elif args.optimizer == 'adamw':
            self.optimizer = optim.AdamW(self.model.parameters(), lr=args.lr_initial, betas=(0.9, 0.999),eps=1e-8, weight_decay=args.weight_decay)
        elif args.optimizer == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=args.lr_initial, momentum=0.9)
         
        ######### Load Pretrained model or Resume train ###########
        if args.pretrained_model is not None:
            ic("Pretrained weight load")
            checkpoint = torch.load(args.pretrained_model)
            try:
                self.model.load_state_dict(checkpoint["model_state"])
            except:
                state_dict = checkpoint["state_dict"]
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:] # remove `module.`
                    new_state_dict[name] = v
                self.model.load_state_dict(new_state_dict)

        if args.resume is not None:
            ic("Loading model and optimizer from checkpoint ")
            checkpoint = torch.load(self.resume_train)
            self.model.load_state_dict(checkpoint["model_state"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
            self.start_epo = checkpoint["epoch"]
            self.logger.info("Loaded checkpoint '{}' (epoch {})".format(self.resume_train, self.start_epo))
            ic(self.start_epo)
        else:
            self.logger.info("No checkpoint found at '{}'".format(self.resume_train))

        ######### Scheduler ###########
        warmup = False
        if warmup:
           warmup_epochs = self.start_epo
           scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.max_epoch-warmup_epochs, eta_min=1e-6)
           self.scheduler = GradualWarmupScheduler(self.optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
           self.scheduler.step()
        else:
            if args.scheduler == 'cosine':
                self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.max_epoch, eta_min=1e-6)
            elif args.scheduler == 'lambda':
                self.scheduler = optim.lr_scheduler.LambdaLR(optimizer=self.optimizer, lr_lambda=lambda epoch: 0.95 ** self.max_epoch, last_epoch=-1)
            
            for i in range(1, self.start_epo):
                self.scheduler.step()
        ic("==> Training with learning rate:", self.scheduler.get_last_lr()[0])

    def criteria_weight(self, out_feature_gt):
        _out_feature = out_feature_gt.detach().numpy().copy()
        category_weight = _out_feature[:, 3, ...]
        confidence_weight = _out_feature[:, 3, ...]
        class_weight = _out_feature[:, 4:5, ...]

        object_idx = np.where(category_weight != 0)
        nonobject_idx = np.where(category_weight == 0)

        category_weight[object_idx] = 2.0
        category_weight[nonobject_idx] = 1.0

        confidence_weight[object_idx] = 2.0
        confidence_weight[nonobject_idx] = 1.0

        object_idx = np.where(class_weight != 0)
        nonobject_idx = np.where(class_weight == 0)
        class_weight[object_idx] = 2.0
        class_weight[nonobject_idx] = 1.0

        category_weight = torch.from_numpy(category_weight).to(self.device)
        confidence_weight = torch.from_numpy(confidence_weight).to(self.device)

        class_weight = np.concatenate(
            [class_weight,
             class_weight,
             class_weight * 15.0,  # bike
             class_weight * 15.0,  # pedestrian
             class_weight], axis=1)
        class_weight = torch.from_numpy(class_weight).to(self.device)
        
        return category_weight, confidence_weight, class_weight

    def result_visualization(self, out_feature_gt, output, in_feature, index, mode, dataloader):
        
        category_gt_img = out_feature_gt[0, 0:1, ...].cpu().detach().numpy().copy()
        confidence_gt_img = out_feature_gt[0, 3:4, ...].cpu().detach().numpy().copy()
        category_img = get_category_or_confidence_image(output[0, 0:1, :, :],self.height, self.width, thresh=0.3)
        confidence_img = get_category_or_confidence_image(output[0, 3:4, :, :],self.height, self.width, thresh=0.3)
        label_img = get_class_image(out_feature_gt[0, 4:10, ...],self.height, self.width)
        pred_class_img = get_class_image(output[0, 4:10, ...],self.height, self.width)
        in_feature_img = in_feature[0,self.non_empty_channle:self.non_empty_channle+1,...].cpu().detach().numpy().copy()
        in_feature_img[in_feature_img > 0] = 255

        instance_gt_image = None
        instance_image = None
        heading_gt_image = None
        heading_image = None
        if np.mod(index, len(dataloader) - 1) == 0 and index != 0:
        #if index != 0:
            instance_gt_image = get_arrow_image(
                in_feature[0, ...].cpu().detach().numpy().transpose(1, 2, 0),
                out_feature_gt[0, ...].cpu().detach().numpy().transpose(1, 2, 0),
                timeout=10, viz_range=1)

            instance_image = get_arrow_image(
                in_feature[0, ...].cpu().detach().numpy().transpose(1, 2, 0),
                output[0, ...].cpu().detach().numpy().transpose(1, 2, 0), 
                timeout=10, viz_range=1)
            
            heading_gt_image = get_arrow_image(
                in_feature[0, ...].cpu().detach().numpy().transpose(1, 2, 0),
                out_feature_gt[0, ...].cpu().detach().numpy().transpose(1, 2, 0),
                draw_target='heading',
                timeout=10, viz_range=1)

            heading_image = get_arrow_image(
                in_feature[0, ...].cpu().detach().numpy().transpose(1, 2, 0),
                output[0, ...].cpu().detach().numpy().transpose(1, 2, 0),
                draw_target='heading',
                timeout=10, viz_range=1)


        self.vis.images(in_feature_img,
                        win='{} in_feature'.format(mode),
                        opts=dict(
                        title='{} in_feature'.format(mode)))
        self.vis.images([category_gt_img, category_img],
                        win='{}_category'.format(mode),
                        opts=dict(
                        title='{} category(GT, Pred)'.format(mode)))
        self.vis.images([confidence_gt_img, confidence_img],
                        win='{}_confidence'.format(mode),
                        opts=dict(
                        title='{} confidence(GT, Pred)'.format(mode)))
        self.vis.images([label_img, pred_class_img],
                        win='{}_class'.format(mode),
                        opts=dict(
                        title='{} class pred(GT, Pred)'.format(mode)))
        if instance_image is not None and instance_gt_image is not None:
            instance_image = instance_image.transpose(2,0,1)
            instance_gt_image = instance_gt_image.transpose(2,0,1)
            self.vis.images([instance_gt_image, instance_image],
                        win='{}_instance (GT, Pred)'.format(mode),
                        opts=dict(title='{} instance (GT, Pred)'.format(mode)))
        if heading_image is not None and heading_gt_image is not None:
            heading_image = heading_image.transpose(2,0,1)
            heading_gt_image = heading_gt_image.transpose(2,0,1)
            self.vis.images([heading_gt_image, heading_image],
                        win='{}_heading (GT, Pred)'.format(mode),
                        opts=dict(title='{} heading (GT, Pred)'.format(mode)))
        
        ### Loss visualazation in graph
        # self.vis.line(X=np.array([self.epo]),
        #             Y=np.array([avg_loss]),
        #             win='loss', name='{}_loss'.format(mode), update='append')
        # self.vis.line(X=np.array([self.epo]),
        #             Y=np.array([avg_category_loss]),
        #             win='loss', name='category_{}_loss'.format(mode),
        #             update='append')
        # self.vis.line(X=np.array([self.epo]),
        #             Y=np.array([avg_confidence_loss]),
        #             win='loss', name='confidence_{}_loss'.format(mode),
        #             update='append')
        # self.vis.line(X=np.array([self.epo]),
        #             Y=np.array([avg_class_loss]),
        #             win='loss', name='class_{}_loss'.format(mode),
        #             update='append')
        # self.vis.line(X=np.array([self.epo]),
        #             Y=np.array([avg_instance_x_loss]),
        #             win='loss', name='instance_x_{}_loss'.format(mode),
        #             update='append')
        # self.vis.line(X=np.array([self.epo]),
        #             Y=np.array([avg_instance_y_loss]),
        #             win='loss', name='instance_y_{}_loss'.format(mode),
        #             update='append')
        # self.vis.line(X=np.array([self.epo]),
        #             Y=np.array([avg_heading_x_loss]),
        #             win='loss', name='heading_x_{}_foss'.format(mode),
        #             update='append')
        # self.vis.line(X=np.array([self.epo]),
        #             Y=np.array([avg_heading_y_loss]),
        #             win='loss', name='heading_y_{}_loss'.format(mode),
        #             update='append')
        # self.vis.line(X=np.array([self.epo]),
        #             Y=np.array([avg_height_loss]),
        #             win='loss', name='height_{}_loss'.format(mode),
        #             update='append')

    def step_train(self, mode):
        ic('Start {} -> epoch: {}'.format(mode,self.epo))
        self.model.train()

        self.best_test = False
        loss_sum = 0
        iter = 0

        loss_sum = 0
        category_loss_sum = 0
        confidence_loss_sum = 0
        class_loss_sum = 0
        instance_x_loss_sum = 0
        instance_y_loss_sum = 0
        heading_x_loss_sum = 0
        heading_y_loss_sum = 0
        height_loss_sum = 0

        ######### Loss ###########
        criterion = get_loss_function('BcnnLoss').to(self.device)        
        
        for index, (in_feature, out_feature_gt) in tqdm.tqdm(enumerate(self.train_dataloader), total=len(self.train_dataloader)):
            iter += 1

            category_weight, confidence_weight, class_weight = self.criteria_weight(out_feature_gt)

            in_feature = in_feature.to(self.device)
            out_feature_gt = out_feature_gt.to(self.device)

            ### Predict image ###            
            output = self.model(in_feature)
        
            (category_loss, confidence_loss, class_loss, instance_x_loss, instance_y_loss, heading_x_loss, heading_y_loss, height_loss) \
                = criterion(output, in_feature, out_feature_gt, category_weight, confidence_weight, class_weight)

            loss = category_loss + confidence_loss + class_loss + (instance_x_loss + instance_y_loss
                   + heading_x_loss + heading_y_loss) * 1.0 + height_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if self.vis_on and np.mod(index, self.vis_interval) == 0:
                self.result_visualization(out_feature_gt, output, in_feature, index, mode, self.train_dataloader)

            loss_for_record = category_loss + confidence_loss \
                + class_loss + instance_x_loss + instance_y_loss \
                + heading_x_loss + heading_y_loss + height_loss
            iter_loss = loss_for_record.item()
            loss_sum += iter_loss
            category_loss_sum += category_loss.item()
            confidence_loss_sum += confidence_loss.item()
            class_loss_sum += class_loss.item()
            instance_x_loss_sum += instance_x_loss.item()
            instance_y_loss_sum += instance_y_loss.item()
            heading_x_loss_sum += heading_x_loss.item()
            heading_y_loss_sum += heading_y_loss.item()
            height_loss_sum += height_loss.item()

            if self.vis_on and np.mod(index, self.vis_interval) == 0:
                self.result_visualization(out_feature_gt, output, in_feature, index, mode, self.train_dataloader)

        self.scheduler.step()
        
        if len(self.train_dataloader) > 0:
            avg_loss = loss_sum / len(self.train_dataloader)
            avg_confidence_loss = confidence_loss_sum / len(self.train_dataloader)
            avg_category_loss = category_loss_sum / len(self.train_dataloader)
            avg_class_loss = class_loss_sum / len(self.train_dataloader)
            avg_instance_x_loss = instance_x_loss_sum / len(self.train_dataloader)
            avg_instance_y_loss = instance_y_loss_sum / len(self.train_dataloader)
            avg_heading_x_loss = heading_x_loss_sum / len(self.train_dataloader)
            avg_heading_y_loss = heading_y_loss_sum / len(self.train_dataloader)
            avg_height_loss = height_loss_sum / len(self.train_dataloader)
        else:
            avg_loss = loss_sum
            avg_confidence_loss = confidence_loss_sum
            avg_category_loss = category_loss_sum
            avg_class_loss = class_loss_sum
            avg_instance_x_loss = instance_x_loss_sum
            avg_instance_y_loss = instance_y_loss_sum
            avg_heading_x_loss = heading_x_loss_sum
            avg_heading_y_loss = heading_y_loss_sum
            avg_height_loss = height_loss_sum

        ### Save latest model ###
        if np.mod(self.epo, self.save_model_interval) == 0:
            _state = {
                "epoch": self.epo + 1,
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
            } 
            _save_path_latest = osp.join(self.logdir, '{}_latestmodel.pth'.format('glare'))
            torch.save(_state,_save_path_latest)
        
        ### Print result of loss and scheduler###
        format_str = "epoch: {}\n total loss: {:3f}, category_loss: {:.3f}, confidence_loss: {:.3f}, class_loss:{:.3f}, instace_x_loss:{:.3f}, instace_y_loss={:.3f}, heading_x_loss={:.3f}, heading_y_loss={:.3f}, height_loss={:.3f}, learning_rate={}"
        print_str = format_str.format(int(self.epo) ,float(avg_loss),float(avg_category_loss),float(avg_confidence_loss),float(avg_class_loss), \
            float(avg_instance_x_loss), float(avg_instance_y_loss), float(avg_heading_x_loss), float(avg_heading_y_loss), float(avg_height_loss), float(self.scheduler.get_last_lr()[0]))
        ic(print_str)
        self.logger.info(print_str)


    def step_val(self, mode):
        self.model.eval()
        
        ic('Start {} -> epoch: {}'.format(mode,self.epo))
        self.best_test = False
        psnr_val_rgb = []

        loss_sum = 0
        category_loss_sum = 0
        confidence_loss_sum = 0
        class_loss_sum = 0
        instance_x_loss_sum = 0
        instance_y_loss_sum = 0
        heading_x_loss_sum = 0
        heading_y_loss_sum = 0
        height_loss_sum = 0

        ######### Loss ###########
        criterion = get_loss_function('BcnnLoss').to(self.device)  

        for index, (in_feature, out_feature_gt) in tqdm.tqdm(enumerate(self.val_dataloader), total=len(self.val_dataloader)):
            category_weight, confidence_weight, class_weight = self.criteria_weight(out_feature_gt)

            in_feature = in_feature.to(self.device)
            out_feature_gt = out_feature_gt.to(self.device)

            with torch.no_grad():
                ###Pred image ###
                output = self.model(in_feature)
                
            (category_loss, confidence_loss, class_loss, instance_x_loss, instance_y_loss, heading_x_loss, heading_y_loss, height_loss) \
                = criterion(output, in_feature, out_feature_gt, category_weight, confidence_weight, class_weight)

            loss_for_record = category_loss + confidence_loss \
                + class_loss + instance_x_loss + instance_y_loss \
                + heading_x_loss + heading_y_loss + height_loss
            iter_loss = loss_for_record.item()
            loss_sum += iter_loss
            category_loss_sum += category_loss.item()
            confidence_loss_sum += confidence_loss.item()
            class_loss_sum += class_loss.item()
            instance_x_loss_sum += instance_x_loss.item()
            instance_y_loss_sum += instance_y_loss.item()
            heading_x_loss_sum += heading_x_loss.item()
            heading_y_loss_sum += heading_y_loss.item()
            height_loss_sum += height_loss.item()

            if self.vis_on and np.mod(index, self.vis_interval) == 0:
                self.result_visualization(out_feature_gt, output, in_feature, index, mode, self.val_dataloader)

        ### Save best model ###
        if self.best_loss > loss_sum / len(self.val_dataloader):
            self.best_loss = loss_sum / len(self.val_dataloader)
            _state = {
                "epoch": self.epo + 1,
                "model_state": self.model.state_dict(),
            } 
            _save_path = osp.join(self.logdir, '{}_bestmodel_{}.pth'.format('glare',str(float(self.best_loss))))
            directory = os.path.dirname(_save_path)
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save(_state, _save_path)

        ### Print result of PSNR###
        if len(self.val_dataloader) > 0:
            avg_loss = loss_sum / len(self.val_dataloader)
            avg_confidence_loss = confidence_loss_sum / len(self.val_dataloader)
            avg_category_loss = category_loss_sum / len(self.val_dataloader)
            avg_class_loss = class_loss_sum / len(self.val_dataloader)
            avg_instance_x_loss = instance_x_loss_sum / len(self.val_dataloader)
            avg_instance_y_loss = instance_y_loss_sum / len(self.val_dataloader)
            avg_heading_x_loss = heading_x_loss_sum / len(self.val_dataloader)
            avg_heading_y_loss = heading_y_loss_sum / len(self.val_dataloader)
            avg_height_loss = height_loss_sum / len(self.val_dataloader)
        else:
            avg_loss = loss_sum
            avg_confidence_loss = confidence_loss_sum
            avg_category_loss = category_loss_sum
            avg_class_loss = class_loss_sum
            avg_instance_x_loss = instance_x_loss_sum
            avg_instance_y_loss = instance_y_loss_sum
            avg_heading_x_loss = heading_x_loss_sum
            avg_heading_y_loss = heading_y_loss_sum
            avg_height_loss = height_loss_sum

        format_str = "epoch: {}\n total loss: {:3f}, category_loss: {:.3f}, confidence_loss: {:.3f}, class_loss:{:.3f}, instace_x_loss:{:.3f}, instace_y_loss={:.3f}, heading_x_loss={:.3f}, heading_y_loss={:.3f}, height_loss={:.3f}, learning_rate={}"
        print_str = format_str.format(int(self.epo) ,float(avg_loss),float(avg_category_loss),float(avg_confidence_loss),float(avg_class_loss), \
                float(avg_instance_x_loss), float(avg_instance_y_loss), float(avg_heading_x_loss), float(avg_heading_y_loss), float(avg_height_loss), float(self.scheduler.get_last_lr()[0]))
        ic(print_str)
        self.logger.info(print_str)

    def train(self):
        """Start training."""
        for self.epo in range(self.start_epo, self.max_epoch):
            self.step_train('train')
            self.step_val('val')

if __name__ == "__main__":
    ic.configureOutput(prefix='Deglare training |')
    ######### parser ###########
    args = options.Options().init(argparse.ArgumentParser(description='image deglare')).parse_args()
    ic(args)

    #ic.enable()
    #ic.disable()
    
    trainer = Trainer(args=args)
    trainer.train()
