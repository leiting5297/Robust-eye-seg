import torch
import torch.nn as nn
import torch.nn.functional as F

############注意力#############
class MA(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(MA, self).__init__()

        self.down = nn.Linear(768,256)
        self.norm  = nn.LayerNorm(768)
        self.up   = nn.Linear(256,768)
    def forward(self, x):
        x1 = self.norm(x)
        x2 = self.norm(x)

        x1 = F.gelu(self.down(x1))
        x2 = F.gelu(self.down(x2))
        new_x = x1 + x2

        new_x = self.up(new_x)
        out = F.gelu(new_x)

        return out
