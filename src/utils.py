import torch
import torch.nn.functional as F
import torch.nn as nn

def save_checkpoint(state, is_best=False, filename='checkpoint.h5'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.h5')

class CrossEntropyLoss2D(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(CrossEntropyLoss2D, self).__init__()
        self.nll_loss = nn.NLLLoss2d(weight, size_average)

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs), targets)
