import torch.nn as nn

class test(nn.Module):
    def __init__(self):
        super().__init__()
        
        #((W-F) + 2P)/S + 1 =>
        self.conv2 = nn.Conv2d(1, 4, kernel_size = 3, stride=2, padding=1,bias=False) #160x160x4
        self.b2 = nn.BatchNorm2d(4)
        self.relu2 = nn.ReLU() 
        self.pool2 = nn.MaxPool2d(2, 2) #80x80x4
        
        self.conv3 = nn.Conv2d(4, 8, kernel_size = 3, stride=1, padding =1,bias=False) #80x80x8
        self.b3 = nn.BatchNorm2d(8)
        self.relu3 = nn.ReLU() 
        self.pool3 = nn.MaxPool2d(2, 2) # 40x40x8
        
        self.conv4 = nn.Conv2d(8, 2, kernel_size = 3, stride=1, padding=1,bias=False) #40x40x2
        self.b4 = nn.BatchNorm2d(2)
        self.relu4 = nn.ReLU() # 40x40x2 
        
    def forward(self, x):
        x = self.conv2(x)
        x = self.b2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.b3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        x = self.conv4(x)
        x = self.b4(x)
        out = self.relu4(x)
        return out