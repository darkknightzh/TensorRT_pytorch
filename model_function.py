from torch import nn

class Example_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model1 = nn.Conv2d(3, 24, 3)
        self.ReLU = nn.ReLU()
        self.model2 = nn.Conv2d(24, 3, 3)

    def forward(self, x):
        out = self.ReLU(self.model1(x))
        out = self.model2(out)
        return out
