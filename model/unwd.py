import torch
import torch.nn as nn
import torch.nn.functional as F
from model.srcnn import SRCNN


class up2(nn.Module):
    def __init__(self):
        super(up2, self).__init__()
        self.up = nn.PixelShuffle(2)

    def forward(self, input):
        return self.up2(input)


class up5(nn.Module):
    def __init__(self):
        super(up5, self).__init__()
        self.up = nn.PixelShuffle(5)

    def forward(self, input):
        return self.up5(input)


class B_Conv(nn.Module):
    def __init__(self, inC, n_feat):
        super(B_Conv, self).__init__()
        self.conv1 = nn.Conv2d(inC, n_feat, 1)
        self.conv3 = nn.Conv2d(inC, n_feat, 3, 1, 1)
        self.conv5 = nn.Conv2d(inC, n_feat, 5, 1, 2)

    def forward(self, x):
        out1 = self.conv1(x)
        out3 = self.conv3(x)
        out5 = self.conv5(x)
        return out1 + out3 + out5


class RDB_Conv(nn.Module):
    def __init__(self, inC, n_feat):
        super(RDB_Conv, self).__init__()
        self.conv = nn.Sequential(*[
            B_Conv(inC, n_feat),
            nn.ReLU()
        ])

    def forward(self, x):
        out = self.conv(x)
        return torch.cat([x, out], 1)


class RDB(nn.Module):
    def __init__(self, inC, n_feat, n_block):
        super(RDB, self).__init__()
        convs = []
        for c in range(n_block):
            convs.append(RDB_Conv(inC + c * n_feat, n_feat))
        self.convs = nn.Sequential(*convs)
        self.LFF = nn.Conv2d(inC + n_block * n_feat, inC, 1, padding=0, stride=1)

    def forward(self, x):
        return self.LFF(self.convs(x)) + x


class SFTLayer(nn.Module):
    def __init__(self):
        super(SFTLayer, self).__init__()
        self.SFT_scale_conv0 = nn.Conv2d(32, 32, 1)
        self.SFT_scale_conv1 = nn.Conv2d(32, 32, 1)
        self.SFT_shift_conv0 = nn.Conv2d(32, 32, 1)
        self.SFT_shift_conv1 = nn.Conv2d(32, 32, 1)

    def forward(self, x):
        scale = self.SFT_scale_conv1(F.leaky_relu(self.SFT_scale_conv0(x[1]), 0.1, inplace=True))
        shift = self.SFT_shift_conv1(F.leaky_relu(self.SFT_shift_conv0(x[1]), 0.1, inplace=True))
        return x[0] * (scale + 1) + shift


class ResBlock_SFT(nn.Module):
    def __init__(self):
        super(ResBlock_SFT, self).__init__()
        self.sft0 = SFTLayer()
        self.conv0 = nn.Conv2d(32, 32, 3, 1, 1)
        self.sft1 = SFTLayer()
        self.conv1 = nn.Conv2d(32, 32, 3, 1, 1)

    def forward(self, x):
        fea = self.sft0(x)
        fea = F.relu(self.conv0(fea), inplace=True)
        fea = self.sft1((fea + x[0], x[1]))
        fea = self.conv1(fea)
        return (x[0] + fea, x[1])


class AGSDNetV2(nn.Module):
    def __init__(self):
        super(AGSDNetV2, self).__init__()
        self.numB = 4
        self.numF = 4

        self.CondNet = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1), nn.LeakyReLU(0.1, True), nn.Conv2d(64, 64, 1),
            nn.LeakyReLU(0.1, True), nn.Conv2d(64, 64, 1), nn.LeakyReLU(0.1, True),
            nn.Conv2d(64, 64, 1), nn.LeakyReLU(0.1, True), nn.Conv2d(64, 32, 1))

        self.inNet = nn.Conv2d(4, 32, kernel_size=3, padding=1)
        self.RDBs = nn.ModuleList()
        for i in range(self.numB):
            self.RDBs.append(RDB(32, 32, 6))

        sft_branch = []
        for i in range(self.numF):
            sft_branch.append(ResBlock_SFT())
        sft_branch.append(SFTLayer())
        sft_branch.append(nn.Conv2d(32, 32, 3, 1, 1))
        self.sft_branch = nn.Sequential(*sft_branch)

        self.HR_branch = nn.Sequential(
            nn.ReLU(), nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(), nn.Conv2d(64, 32, 3, 1, 1))

        self.up1 = nn.Sequential(
            nn.ReLU(), nn.Conv2d(32, 32 * 2 ** 2, 3, 1, 1), nn.PixelShuffle(2))
        self.up2 = nn.Sequential(
            nn.ReLU(), nn.Conv2d(32, 32 * 5 ** 2, 3, 1, 1), nn.PixelShuffle(5))
        self.HR_out = nn.Conv2d(32, 1, 3, 1, 1)

        self.srcnn = SRCNN(4)
        self.aux = SRCNN(2)


    def forward(self, x, dem, xb, xn, dem2):
        df = self.CondNet(dem)
        dx = self.inNet(x)
        for i in range(self.numB):
            dx = self.RDBs[i](dx)
        x = self.sft_branch((dx, df))
        x = self.HR_branch(x)

        x1 = self.up1(x)
        x2 = self.up2(x1)
        x1 = self.HR_out(x1)
        x2 = self.HR_out(x2)

        x = torch.cat([x2, xb, xn, dem2], 1)
        x = self.srcnn(x)
        y = self.aux(torch.cat([x2, dem2], 1))
        return x, y


if __name__ == '__main__':
    device = torch.device('cuda:7')
    model = AGSDNetV2().to(device).to(device)
    input = torch.randn((1,4,15,14)).float().to(device)
    dem = torch.randn((1,1,15,14)).float().to(device)
    dem2 = torch.randn((1,1,150,140)).float().to(device)
    output = model(input, dem, dem2, dem2, dem2)