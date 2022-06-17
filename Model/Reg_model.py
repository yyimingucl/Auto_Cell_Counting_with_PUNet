import torch
from torch.nn import (Module,Conv2d, ReLU, MaxPool2d, Linear, Sequential)
import torchvision.models as models
from torch import hub
from torchvision.models import resnet50, ResNet50_Weights
from torch.nn.init import (xavier_normal_, kaiming_normal_)

class L2(torch.nn.Module):
    def __init__(self, module, weight_decay):
        super().__init__()
        self.module = module
        self.weight_decay = weight_decay

        # Backward hook is registered on the specified module
        self.hook = self.module.register_full_backward_hook(self._weight_decay_hook)

    # Not dependent on backprop incoming values, placeholder
    def _weight_decay_hook(self, *_):
        for param in self.module.parameters():
            # If there is no gradient or it was zeroed out
            # Zeroed out using optimizer.zero_grad() usually
            # Turn on if needed with grad accumulation/more safer way
            # if param.grad is None or torch.all(param.grad == 0.0):

            # Apply regularization on it
            param.grad = self.regularize(param)

    def regularize(self, parameter):
        # L1 regularization formula
        return 2 * self.weight_decay * parameter.data

    def forward(self, *args, **kwargs):
        # Simply forward and args and kwargs to module
        return self.module(*args, **kwargs)


                       
def Modified_Resnet50():
    # model = hub.load('pytorch/vision:v0.10.0', 'resnet50', weight=ResNet50_Weights)
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    l1 = Linear(in_features=2048, out_features=512, bias=True)
    l2 = Linear(in_features=512, out_features=128, bias=True)
    l3 = Linear(in_features=128, out_features=6, bias=True)
    
    xavier_normal_(l1.weight)
    xavier_normal_(l2.weight)
    xavier_normal_(l3.weight)
    
    model.fc = l1

    modified_resnet50 = Sequential(model,
                                   ReLU(),
                                   l2,
                                   ReLU(),
                                   L2(l3, weight_decay=0.2),
                                   ReLU())
    return modified_resnet50