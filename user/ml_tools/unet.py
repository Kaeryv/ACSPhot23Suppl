import torch
import torch.nn.functional as F
import torch.nn as nn

def load(state_dict_path, device="cpu", scalers=False, model=True, complexity=0):
    model_data = torch.load(state_dict_path)
    
    if model:
        model = UNet(complexity=3)#model_data["complexity"]
        model.load_state_dict(model_data["model"])
        model.eval()
        dev = torch.device(device)
        model.to(dev)

        if scalers:
            return model, model_data["xscaler"], model_data["yscaler"]
        else:
            return model
    elif scalers:
        return model_data["xscaler"], model_data["yscaler"]


def dual_conv2d(in_channels, out_channels, ks=3, pad=1):
    return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=ks, padding=pad, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=ks, padding=pad, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
    )

def encoding_block(in_channels, out_channels):
    return nn.Sequential(
            nn.MaxPool2d(2),
            dual_conv2d(in_channels, out_channels)
    )

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = dual_conv2d(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffX = x2.size()[3] - x1.size()[3]
        diffY = x2.size()[2] - x1.size()[2]

        x1 = F.pad(x1, [ diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)

        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, complexity=16):
        super(UNet, self).__init__()
        c = complexity
        self.c1 = dual_conv2d(1, 2*c)
        self.e1 = encoding_block(2*c, 4*c)
        self.e2 = encoding_block(4*c, 8*c)
        self.e3 = encoding_block(8*c, 16*c)

        self.u1 = DecoderBlock(16*c, 8*c)
        self.u2 = DecoderBlock(8*c, 4*c)
        self.u3 = DecoderBlock(4*c, 2*c)
        self.c2 = nn.Conv2d(2*c, 1, kernel_size=1)

    def forward(self, x):
        x1 = self.c1(x)
        x2 = self.e1(x1)
        x3 = self.e2(x2)
        x4 = self.e3(x3)

        x = self.u1(x4, x3)
        x = self.u2(x,  x2)
        x = self.u3(x,  x1)

        return self.c2(x)

