#!/usr/bin/env python3
# coding: utf-8

import argparse
import os.path as osp
import sys

import gdown
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import tqdm
import visdom
from datetime import datetime

sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))

from dataset.NuscData import load_dataset
from optimizers import get_optimizer
from schedulers import get_scheduler
from loss import get_loss_function
from models.BCNN import BCNN

from utils.visualize_utils import get_arrow_image, get_class_image, get_category_or_confidence_image, get_input_feature_image  # noqa
from utils.utils import get_logger
from icecream import ic



def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight)

class Trainer(object):
    """CNN trainer.

    Parameters
    ----------
    data_path : str
        Training data path.
    batch_size : int
    max_epoch : int
    pretrained_model : str
        Pretrained model path.
    train_data_num : int
        Number of data used for training.
        Larger number if all data are used.
    val_data_num : int
        Number of data used for validation.
        Larger number if all data are used.
    width : int
        feature map width.
    height :int
        feature map height.
    use_constant_feature : bool
        Whether to use constant feature.
    use_intensity_feature : bool
        Whether to use intensity feature.

    """

    def __init__(self, data_path, batch_size, max_epoch, pretrained_model,
                 train_data_num, val_data_num,
                 width, height, use_constant_feature, use_intensity_feature, vis_on, resume_train, work_dir):

        self.train_dataloader, self.val_dataloader \
            = load_dataset(data_path, batch_size)
        self.max_epoch = max_epoch
        self.time_now = datetime.now().strftime('%Y%m%d_%H%M')
        self.best_loss = 1e10
        self.vis_on = vis_on
        if self.vis_on:
            self.vis = visdom.Visdom()
            self.vis_interval = 1

        self.logdir = osp.join("./", work_dir)
        self.logger = get_logger(self.logdir)
        self.logger.info("Let the train begin...")

        if use_constant_feature and use_intensity_feature:
            self.in_channels = 8
            self.non_empty_channle = 7
        elif use_constant_feature or use_intensity_feature:
            self.in_channels = 6
            self.non_empty_channle = 5
        else:
            self.in_channels = 4
            self.non_empty_channle = 3

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = BCNN(in_channels=self.in_channels, n_class=5).to(self.device)
        self.model = torch.nn.DataParallel(self.model, device_ids=range(torch.cuda.device_count()))  # multi gpu
        self.model.apply(weights_init)

        self.save_model_interval = 1
        self.loss_print_interval = 1

        self.optimizer = get_optimizer("SGD", self.model)
        self.scheduler = get_scheduler('LambdaLR', self.optimizer, self.max_epoch)

        total_params = sum(p.numel() for p in self.model.parameters())
        #print( 'Parameters:',total_params )
        ic('Parameters:',total_params)

        self.start_epo = 0
        if resume_train is not None:
            self.logger.info("Loading model and optimizer from checkpoint ")
            #print("Loading model and optimizer from checkpoint ")
            ic("Loading model and optimizer from checkpoint ")
            checkpoint = torch.load(resume_train)
            self.model.load_state_dict(checkpoint["model_state"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
            self.scheduler.load_state_dict(checkpoint["scheduler_state"])
            self.start_epo = checkpoint["scheduler_state"]["last_epoch"]
            self.logger.info("Loaded checkpoint '{}' (epoch {})".format(resume_train, self.start_epo))
        else:
            self.logger.info("No checkpoint found at '{}'".format(resume_train))

        self.train_data_num = train_data_num
        self.val_data_num = val_data_num

        self.width = width
        self.height = height
        self.grid_range = 90.

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

    def step(self, mode):
        """Proceed with training or verification

        Parameters
        ----------
        mode : str
            Specify training or verification. 'train' or 'val'

        """
        #print('Start {}'.format(mode))
        ic('Start {}'.format(mode))

        if mode == 'train':
            self.model.train()
            dataloader = self.train_dataloader
        elif mode == 'val':
            self.model.eval()
            dataloader = self.val_dataloader

        loss_sum = 0
        category_loss_sum = 0
        confidence_loss_sum = 0
        class_loss_sum = 0
        instance_x_loss_sum = 0
        instance_y_loss_sum = 0
        heading_x_loss_sum = 0
        heading_y_loss_sum = 0
        height_loss_sum = 0

        criterion = get_loss_function('BcnnLoss').to(self.device)
        for index, (in_feature, out_feature_gt) in tqdm.tqdm(enumerate(dataloader), total=len(dataloader),
                desc='{} epoch={}'.format(mode, self.epo), leave=True):

            category_weight, confidence_weight, class_weight = self.criteria_weight(out_feature_gt)
            

            criterion = criterion.to(self.device)
            #creterion = get_loss_function
            #criterion = BcnnLossNew().to(self.device)
            in_feature = in_feature.to(self.device)
            out_feature_gt = out_feature_gt.to(self.device)

            if mode == 'train':
                output = self.model(in_feature)
            elif mode == 'val':
                with torch.no_grad():
                    output = self.model(in_feature)

            (category_loss, confidence_loss, class_loss, instance_x_loss, instance_y_loss, heading_x_loss, heading_y_loss, height_loss) \
                = criterion(output, in_feature, out_feature_gt, category_weight, confidence_weight, class_weight)

            loss = category_loss + confidence_loss + class_loss + (instance_x_loss + instance_y_loss
                   + heading_x_loss + heading_y_loss) * 1.0 + height_loss

            if mode == 'train':
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if self.vis_on and np.mod(index, self.vis_interval) == 0:
                self.result_visualization(out_feature_gt, output, in_feature, index, mode, dataloader)

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

            if mode == 'train':
                if index == self.train_data_num - 1:
                    #print("Finish training {} data.".format(index))
                    ic("Finish training {} data.".format(index))
                    break
            elif mode == 'val':
                if index == self.val_data_num - 1:
                    #print("Finish validating {} data".format(index))
                    ic("Finish validating {} data".format(index))
                    break

        if len(dataloader) > 0:
            avg_loss = loss_sum / len(dataloader)
            avg_confidence_loss = confidence_loss_sum / len(dataloader)
            avg_category_loss = category_loss_sum / len(dataloader)
            avg_class_loss = class_loss_sum / len(dataloader)
            avg_instance_x_loss = instance_x_loss_sum / len(dataloader)
            avg_instance_y_loss = instance_y_loss_sum / len(dataloader)
            avg_heading_x_loss = heading_x_loss_sum / len(dataloader)
            avg_heading_y_loss = heading_y_loss_sum / len(dataloader)
            avg_height_loss = height_loss_sum / len(dataloader)
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
        
        if np.mod(index, self.loss_print_interval) == 0:
            format_str = "epoch: {}\n total loss: {:3f}, category_loss: {:.3f}, confidence_loss: {:.3f}, class_loss:{:.3f}, instace_x_loss:{:.3f}, instace_y_loss={:.3f}, heading_x_loss={:.3f}, heading_y_loss={:.3f}, height_loss={:.3f}"
            print_str = format_str.format(int(self.epo) ,float(avg_loss),float(avg_category_loss),float(avg_confidence_loss),float(avg_class_loss), \
                float(avg_instance_x_loss), float(avg_instance_y_loss), float(avg_heading_x_loss), float(avg_heading_y_loss), float(avg_height_loss))
            ic(print_str)
            self.logger.info(print_str)

        if mode == 'val':
            _state = {
                "epoch": index + 1,
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "scheduler_state": self.scheduler.state_dict(),
            } 
            _save_path = osp.join(self.logdir, '{}_latestmodel.pt'.format('bcnn'))
            if np.mod(self.epo, self.save_model_interval) == 0:
                torch.save(_state,_save_path)

            if self.best_loss > loss_sum / len(dataloader):
                #print('update best model {} -> {}'.format(self.best_loss, loss_sum / len(dataloader)))
                ic('update best model {} -> {}'.format(self.best_loss, loss_sum / len(dataloader)))
                self.best_loss = loss_sum / len(dataloader)
                _save_path = osp.join(self.logdir, '{}_bestmodel.pt'.format('bcnn'))
                torch.save(_state, _save_path)

    def train(self):
        """Start training."""
        for self.epo in range(self.start_epo, self.max_epoch):
            self.step('train')
            self.step('val')
            self.scheduler.step()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--data_path',
        '-dp',
        type=str,
        help='Training data path',
        #default='/dataset/nuScenes/FeatureExtracted/v1.0-trainval/')
        default='/dataset/nuScenes/FeatureExtracted2/v1.0-mini/')
    parser.add_argument('--batch_size', '-bs', type=int,
                        help='batch size',
                        default=5)
    parser.add_argument('--max_epoch', '-me', type=int,
                        help='max epoch',       
                        default=300)
    parser.add_argument('--pretrained_model', '-p', type=str,
                        help='Pretrained model path',
                        default='./checkpoints/bcnn_latestmodel_20210607_0809.pt')
    parser.add_argument('--train_data_num', '-tn', type=int,
                        help='Number of data used for training. Larger number if all data are used.',
                        default=1000000)
    parser.add_argument('--val_data_num', '-vn', type=int,
                        help='Nuber of  data used for validation. Larger number if all data are used.',
                        default=100000)
    parser.add_argument('--width', type=int,
                        help='feature map width',
                        default=864)
    parser.add_argument('--height', type=int,
                        help='feature map height',
                        default=864)
    parser.add_argument('--use_constant_feature', type=int,
                        help='Whether to use constant feature',
                        default=0)
    parser.add_argument('--use_intensity_feature', type=int,
                        help='Whether to use intensity feature',
                        default=0)
    parser.add_argument('--visualization_on', type=int,
                        help='Whether to use visualaziation on during train',
                        default=1)
    parser.add_argument('--resume', type=str,
                        help='Train process resume cur/bcnn_latestmodel.pt',
                        default='./cur/bcnn/bcnn_latestmodel_0628.pt')
                        #default=None)
    parser.add_argument('--work_dir', type=str,
                        help='Work directory cur/bcnn',
                        default='./cur/bcnn')

    args = parser.parse_args()
    
    ic.configureOutput(prefix='CNN training |')
    ic.enable()
    #ic.disable()

    trainer = Trainer(data_path=args.data_path,
                      batch_size=args.batch_size,
                      max_epoch=args.max_epoch,
                      pretrained_model=args.pretrained_model,
                      train_data_num=args.train_data_num,
                      val_data_num=args.val_data_num,
                      width=args.width,
                      height=args.height,
                      use_constant_feature=args.use_constant_feature,
                      use_intensity_feature=args.use_intensity_feature,
                      vis_on = args.visualization_on,
                      resume_train = args.resume,
                      work_dir = args.work_dir)

    trainer.train()
