from loss.BcnnLoss import BcnnLoss
from loss.BcnnLossNew import BcnnLossNew

def get_loss_function(loss_function_type):
    if loss_function_type == 'BcnnLoss':
        return BcnnLoss()
    elif loss_function_type == 'BcnnLossNew':
	    return BcnnLossNew()
    else:
	    raise NotImplementedError("Loss {} not implemented".format(loss_function_type))
	    return 0
