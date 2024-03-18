import torch
import torch.nn as nn

class Dc_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, normalize=True):
        super().__init__()

        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2))
    
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        x = self.block(x)
        return x

#208X46 image(H,W)
class Dcnator(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()

        self.stage_1 = Dc_block(in_channels*2,64,kernel_size=(4,3), stride=(2,1), padding=(1,0),normalize=False)
        self.stage_2 = Dc_block(64,128,kernel_size=(3,3), stride=(2,2), padding=(1,1))
        self.stage_3 = Dc_block(128,256,kernel_size=(3,3), stride=(2,2), padding=(1,1))
        self.stage_4 = Dc_block(256,512,kernel_size=(3,3), stride=2, padding=(1,1))

        self.patch = nn.Conv2d(512,out_channels= 1, kernel_size=(3,3), stride=(2,2), padding=(0,1)) # 78x24 패치 생성

    def forward(self,a,b):
        x = torch.cat((a,b),1)
        x = self.stage_1(x)
        x = self.stage_2(x)
        x = self.stage_3(x)
        x = self.stage_4(x)
        x = self.patch(x)
        x = torch.sigmoid(x)
        return x

