import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, normalize=True, dropout=0.0):
        super().__init__()

        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)]

        if normalize:
            layers.append(nn.InstanceNorm2d(out_channels)),

        layers.append(nn.LeakyReLU(0.2))

        if dropout:
            layers.append(nn.Dropout(dropout))

        self.down = nn.Sequential(*layers)

    def forward(self, x):
        x = self.down(x)
        return x

class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dropout=0.0):
        super().__init__()

        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU()
        ]

        if dropout:
            layers.append(nn.Dropout(dropout))

        self.up = nn.Sequential(*layers)

    def forward(self,x,skip):
        x = self.up(x)
        x = torch.cat((x,skip),1)
        return x

class Generator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()

        self.down1 = Encoder(in_channels=3, out_channels =64, kernel_size =(4,3), stride=(2,1), padding=(1,0), normalize=False)
        self.down2 = Encoder(in_channels=64, out_channels =128, kernel_size =(3,3), stride=(2,2), padding =1)                 
        self.down3 = Encoder(in_channels=128, out_channels =256, kernel_size =(3,3), stride=(2,2), padding =1)               
        self.down4 = Encoder(in_channels=256, out_channels =256, kernel_size =(3,3), stride=(2,2), padding =1, dropout=0.5)
        self.down5 = Encoder(in_channels=256, out_channels =256, kernel_size =(3,3), stride=(2,2), padding =(0,1), dropout=0.5)              
        self.down6 = Encoder(in_channels=256, out_channels =256, kernel_size =(2,2), stride=(2,1), padding =0, normalize=False,dropout=0.5)

        self.up1 = Decoder(in_channels=256, out_channels=256, kernel_size =(2,2), stride=(2,1), padding =0,dropout=0.5)
        self.up2 = Decoder(in_channels=512, out_channels=256, kernel_size =(3,3), stride=(2,2), padding =(0,1),dropout=0.5)
        self.up3 = Decoder(in_channels=512, out_channels=256, kernel_size =(3,3), stride=(2,2), padding =1,dropout=0.5)
        self.up4 = Decoder(in_channels=512, out_channels=128, kernel_size =(3,3), stride=(2,2), padding =1,dropout=0.5)
        self.up5 = Decoder(in_channels=256, out_channels=64, kernel_size =(3,3), stride=(2,2), padding =1,dropout=0.5)
        self.up6 = nn.Sequential(nn.ConvTranspose2d(in_channels=128, out_channels=3, kernel_size =(4,3), stride=(2,1), padding =(1,0)), nn.Tanh())

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        u1 = self.up1(d6,d5)
        u2 = self.up2(u1,d4)
        u3 = self.up3(u2,d3)
        u4 = self.up4(u3,d2)
        u5 = self.up5(u4,d1)
        u6 = self.up6(u5)

        return u6

