import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class MyLoss(nn.Module):
    def __init__(self, mu):
        super(MyLoss, self).__init__()
        self.L1_Loss = nn.L1Loss()

    def forward(self, output, label):
        l1_loss = self.L1_Loss(output, label)
        return l1_loss


