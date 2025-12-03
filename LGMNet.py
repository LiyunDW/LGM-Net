import torch
from torch import nn
import torch.nn.functional as F

PhiT_y, Phi_weight = None, None


class RB(nn.Module):
    def __init__(self, dim):
        super(RB, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 3, padding=1)
        )

    def forward(self, x):
        return x + self.body(x)


class FDIM_layer(nn.Module):
    def __init__(self, dim, Phi_and_PhiT, r, dimf):
        super(FDIM_layer, self).__init__()
        self.r = r
        self.Phi, self.PhiT = Phi_and_PhiT
        self.z = RB(dimf)
        self.G = nn.Sequential(
            nn.Conv2d(r * r + 3 * dimf, dimf, 3, padding=1),
            nn.Conv2d(dimf, dimf, 1)
        )
        self.init = nn.Sequential(
            RB(dimf),
            RB(dimf)
        )
        self.body = nn.Sequential(
            nn.Conv2d(dimf, dim, 1),
            RB(dim),
            RB(dim),
            nn.Conv2d(dim, dimf, 1)
        )

    def forward(self, x):
        global PhiT_y
        if not isinstance(x, tuple):
            x, z = torch.chunk(x, chunks=2, dim=1)
        else:
            x, z = x
        # AEGR
        x_input, z = x, self.z(z)
        x = F.pixel_shuffle(x, self.r)
        b, c, h, w = x.shape
        PhiTPhix = F.pixel_unshuffle(self.PhiT(self.Phi(x.reshape(-1, 1, h, w))).reshape(b, c, h, w), self.r)
        I = torch.cat([x_input, PhiTPhix, z, F.pixel_unshuffle(PhiT_y.to(x.device), self.r)], dim=1)
        I_tlide = self.G(I)
        x = I_tlide + self.init(I_tlide)
        # NSM
        z = x + self.body(x)
        return x, z


class AVF_D(nn.Module):
    def __init__(self, dim):
        super(AVF_D, self).__init__()
        self.pool = nn.MaxPool2d(2, stride=2)
        self.conv = nn.Conv2d(dim, dim * 4, 2, stride=2)
        self.conv3 = nn.Conv2d(dim, dim * 4, 4, stride=2, padding=1)
        self.conv5 = nn.Conv2d(dim, dim * 4, 8, stride=2, padding=3)
        self.merge = nn.Conv2d(dim * 13, dim * 8, 1)

    def forward(self, x):
        x1 = self.pool(x)
        x2 = self.conv(x)
        x3 = self.conv3(x)
        x4 = self.conv5(x)
        x = torch.cat([x1, x2, x3, x4], dim=1)
        x, z = torch.chunk(self.merge(x), 2, dim=1)
        return x, z


class UNet_sub_layer(nn.Module):
    def __init__(self, Iter_D, dim, Phi_and_PhiT, r, dimf, mode=None):
        super(UNet_sub_layer, self).__init__()
        self.mode = mode
        if self.mode == "down":
            self.hqsD = nn.Sequential(
                *[FDIM_layer(dim, Phi_and_PhiT, r, dimf) for _ in range(Iter_D)]
            )
            self.hqsD1 = AVF_D(dimf)
        if self.mode is None:
            self.hqsM = nn.Sequential(
                *[FDIM_layer(dim, Phi_and_PhiT, r, dimf) for _ in range(2 * Iter_D)]
            )
        if self.mode == "up":
            self.hqsU = nn.Sequential(
                nn.ConvTranspose2d(dimf * 4, dimf * 2, 2, stride=2)
            )  # AVF_U
            self.hqsU1 = nn.Sequential(
                *[FDIM_layer(dim, Phi_and_PhiT, r, dimf) for _ in range(Iter_D)]
            )

    def forward(self, x, z):
        if self.mode == "down":
            x, z = self.hqsD(torch.cat([x, z], dim=1))
            x, z = self.hqsD1(x + z)  # AVF_U
        if self.mode is None:
            x, z = self.hqsM(torch.cat([x, z], dim=1))
        if self.mode == "up":
            x, z = torch.chunk(self.hqsU(x + z), 2, dim=1)
            x, z = self.hqsU1(torch.cat([x, z], dim=1))
        return x, z


class UNet(nn.Module):
    def __init__(self, Iter_D, Phi_and_PhiT, dim, dimf):
        super(UNet, self).__init__()
        self.conv_first = nn.Conv2d(2, dimf * 2, 3, padding=1)
        self.down1 = UNet_sub_layer(Iter_D, dim, Phi_and_PhiT, 1, dimf, 'down')
        self.down2 = UNet_sub_layer(Iter_D, 4 * dim, Phi_and_PhiT, 2, dimf * 4, 'down')
        self.mid = UNet_sub_layer(Iter_D, 16 * dim, Phi_and_PhiT, 4, dimf * 16)
        self.up2 = UNet_sub_layer(Iter_D, 4 * dim, Phi_and_PhiT, 2, dimf * 4, 'up')
        self.up1 = UNet_sub_layer(Iter_D, dim, Phi_and_PhiT, 1, dimf, 'up')
        self.kd = nn.Sequential(
            RB(dimf),
            RB(dimf)
        )
        self.conv_last = nn.Conv2d(dimf, 1, 3, padding=1)

    def forward(self, x, z):
        x0, z0 = torch.chunk(self.conv_first(torch.cat([x, z], dim=1)), 2, dim=1)
        x1, z1 = self.down1(x0, z0)
        x2, z2 = self.down2(x1, z1)
        x3, z3 = self.mid(x2, z2)
        x, z = self.up2(x3, z3)
        x, z = self.up1(x + x1, z + z1)
        x = self.kd(x)
        x = self.conv_last(x + x0 + z)
        return x


class LGMNet(nn.Module):
    def __init__(self, Iter_D, Block, Init_Phi, dim, dimf):
        super(LGMNet, self).__init__()
        self.Phi_weight = nn.Parameter(Init_Phi.view(-1, 1, Block, Block))
        self.Phi = lambda x: F.conv2d(x, Phi_weight.to(x.device), stride=Block)
        self.PhiT = lambda x: F.conv_transpose2d(x, Phi_weight.to(x.device), stride=Block)
        self.body = UNet(Iter_D, [self.Phi, self.PhiT], dim, dimf)
        self.z0 = nn.Conv2d(1, 1, 3, padding=1)

    def forward(self, x):
        global PhiT_y, Phi_weight
        Phi_weight = self.Phi_weight.to(x.device)
        y = self.Phi(x)
        x = self.PhiT(y)
        PhiT_y = x.clone()
        z = self.z0(x)
        x = self.body(x, z)
        return x


if __name__ == '__main__':
    from thop import profile, clever_format

    Init_Phi = torch.nn.init.xavier_normal_(torch.Tensor(512, 1024))
    model = LGMNet(4, 32, Init_Phi, 8, 8)
    net_params = sum([p.numel() for p in model.parameters()]) - model.Phi_weight.numel()
    print("total para num: %d" % net_params)

    model.eval()
    macs, params = profile(model, inputs=(torch.randn(1, 1, 256, 256),))
    flops, _ = clever_format([2 * macs, params], '%.2f')
    print(f"FLOPs: {flops}")
