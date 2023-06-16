import inspect

class HyperParameters:
    def save_hyperparameters(self, ignore=[]):
        """Save function arguments into class attributes"""
        frame = inspect.currentframe().f_back
        _, _, _, local_vars = inspect.getargvalues(frame)
        self.hparams = {k:v for k, v in local_vars.items()
                        if k not in set(ignore+['self']) and not k.startswith('_')}
        for k, v in self.hparams.items():
            setattr(self, k, v)
            
class SGD(HyperParameters):
    """Minibatch stochastic gradient descent."""
    def __init__(self, params, lr):
        self.save_hyperparameters()

    def step(self):
        """That is fundamentally all learning amounts to in the end, namely modifying parameters
        of our hypothesis function (in this case linear with as many parameters as features)
        such that the loss function is minimized"""
        for param in self.params:
            param -= self.lr * param.grad  

    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()