from torch.optim.lr_scheduler import LambdaLR, MultiStepLR, ExponentialLR, CosineAnnealingLR

def get_scheduler(scheduler_type, optimizer, epoch):
    if scheduler_type == 'LambdaLR':
        return LambdaLR(optimizer, lr_lambda=lambda epo: 0.9 ** epoch)
    else:
	    raise NotImplementedError("Scheduler {} not implemented".format(scheduler_type))
	    return 0
