from torch.optim import SGD, Adam, ASGD, Adamax, Adadelta, Adagrad, RMSprop

def get_optimizer(optimizer_type, model):
    if optimizer_type == 'SGD':
        return SGD(model.parameters(), lr=2e-6, momentum=0.5, weight_decay=1e-5)
    elif optimizer_type == 'Adam':
	    return Adam(model.parameters(), lr=1e-3)
    else:
	    raise NotImplementedError("Optimizer {} not implemented".format(optimizer_type))
	    return 0
