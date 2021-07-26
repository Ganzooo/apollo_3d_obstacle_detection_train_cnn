import os
import torch
class Options():
    """docstring for Options"""
    def __init__(self):
        pass

    def init(self, parser): 
        #parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        ### General settings:
        parser.add_argument('--data_path', '-dp', type=str, help='Training data path',
                            default='/dataset/nuScenes/FeatureExtracted2/v1.0-mini/')
        parser.add_argument('--batch_size', '-bs', type=int, help='batch size',
                            default=2)
        parser.add_argument('--max_epoch', '-me', type=int, help='max epoch',       
                            default=150)
        parser.add_argument('--pretrained_model', '-p', type=str, help='Pretrained model path',
                            default='./checkpoints/apollo_baseline_unet_weight_model.pth')
                            #default=None)
        parser.add_argument('--width', type=int, help='feature map width',
                            default=864)
        parser.add_argument('--height', type=int, help='feature map height',
                            default=864)
        parser.add_argument('--resume', type=str, help='Train process resume cur/bcnn_latestmodel.pt',
                            #default='/workspace/NETWORK/camera_lens_glare/checkpoints/glare_bestmodel_25.205883026123047.pth')
                            #default= './cur/deglare_act/glare_latestmodel.pth')
                            default=None)
        parser.add_argument('--distributed', type=bool,help='Distributed training mode', default=False)
        
        ### Apex settings
        parser.add_argument('--opt-level', type=str, default = 'O1')
        parser.add_argument('--keep-batchnorm-fp32', type=str, default=None)
        parser.add_argument('--loss-scale', type=str, default=None)
        
        ###Log info setting
        parser.add_argument('--vis_on', type=int, help='vis_on', default=0)
        parser.add_argument('--result_dir', type=str, help='Result directory of savem images', default='./result/')
        parser.add_argument('--work_dir', type=str, help='Work directory cur/bcnn', default='./cur/bcnn')
        parser.add_argument('--save_model_interval', type=int, help='Save model interval', default=1)
        parser.add_argument('--loss_print_interval', type=int, help='Loss print interval', default=1)
        
        
        ### Train settings
        parser.add_argument('--model_type', type=str, help='unet, resnet_unet', default='unet')
        parser.add_argument('--loss_type', type=str, help='BcnnLoss, BcnnLossNew', default='BcnnLoss')
        parser.add_argument('--train_workers', type=int, help='train_dataloader workers', default=16)
        parser.add_argument('--eval_workers', type=int, help='eval_dataloader workers', default=8)
        parser.add_argument('--dataset', type=str, default ='glare_512', help='Dataset type: glare_512, ')
        parser.add_argument('--optimizer', type=str,  help='optimizer for training adamw, adam, sgd', default ='adamw')
        parser.add_argument('--scheduler', type=str,  help='optimizer for training cosine, lambda', default ='cosine')
        parser.add_argument('--lr_initial', type=float, help='initial learning rate', default=0.02)
        parser.add_argument('--weight_decay', type=float, help='weight decay', default=0.001)
        parser.add_argument('--wandb', type=bool, help='use wandb', default=False)
        return parser