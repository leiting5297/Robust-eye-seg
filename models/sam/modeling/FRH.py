import torch
import torch.nn as nn
import torch.nn.functional as F
############特征细化头#############

class FRH(nn.Module):
    def __init__(self, dim):
        super(FRH, self).__init__()

        self.norm = nn.LayerNorm(768)
        self.down = nn.Linear(768,256)
        self.down1 = nn.Linear(256,128)

        self.up1  = nn.Linear(128,256)
        self.up   = nn.Linear(256,768)

    def forward(self, x):
        x1 = self.norm(x)
        x1 = F.gelu(self.down(x1))
        x2 = F.gelu(self.down1(x1))

        x3 = self.up1(x2)
        x3 = self.up(x3)

        out = x + x3
        out = F.gelu(out)
        return out
