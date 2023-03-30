'''
    MSDFFN for Change Detection: Multi-Scale Diff-changed Feature Fusion Network
    Dataset: Farmland(yancheng), River, Hermiston
    author: zhouty
    time:2023.3.30
'''
import torch
import torch.nn as nn

'''
一。 temporal feature encoder–decoder (TFED) subnetworks
'''
class ReducedInception(nn.Module):
    '''
        reduced inception (RI)
    '''
    def __init__(self, in_ch, out_ch, di=4):
        super(ReducedInception, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch//di, kernel_size=1),
            nn.Conv2d(in_ch//di, in_ch//di, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_ch//di),
            nn.ReLU(inplace=True)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch//di, kernel_size=1),
            nn.Conv2d(in_ch//di, in_ch//di, kernel_size=(3,1), padding=(1,0)),
            nn.BatchNorm2d(in_ch//di),
            nn.ReLU(inplace=True)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch//di, kernel_size=1),
            nn.Conv2d(in_ch//di, in_ch//di, kernel_size=(1,3), padding=(0,1)),
            nn.BatchNorm2d(in_ch//di),
            nn.ReLU(inplace=True)
        )
        self.branch4 = nn.Conv2d(in_ch, in_ch // di, kernel_size=1)

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=1),
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        cat = torch.cat((branch1, branch2, branch3, branch4), dim=1)
        out = self.conv(cat)

        return out


class Unet_Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Unet_Down, self).__init__()

        self.RI = ReducedInception(in_ch, out_ch)
        self.Down = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # reduced inception (RI)
        x_i = self.RI(x)
        out = self.Down(x_i)
        return out


class Unet_Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Unet_Up, self).__init__()
        self.Up = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=3),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.Up(x)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=4):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        out = self.sigmoid(out)
        out = out * x
        return out


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(out)
        out = self.sigmoid(out)
        out = out * x
        return out


class SLA(nn.Module):
    '''
        skip layer attention (SLA)
    '''
    def __init__(self, in_ch, out_ch):
        super(SLA, self).__init__()
        self.ca = ChannelAttention(out_ch)
        self.sa = SpatialAttention(out_ch)
        self.convc = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)

    def forward(self, x, f):
        if x.shape[2] == 5:
            x_an = self.ca(x)
        elif x.shape[2] == 7:
            x_an = self.sa(self.ca(x))
        elif x.shape[2] == 9:
            x_an = self.sa(self.ca(x))

        cat = torch.cat([x_an, f], 1)
        out = self.convc(cat)

        return out


class NestedUNet(nn.Module):
    '''
        TFED subnetwork
    '''
    def __init__(self, in_ch=198):
        super().__init__()
        self.conv0_0 = nn.Conv2d(in_ch, 128, kernel_size=1)
        self.down0_1 = Unet_Down(128, 256)
        self.down1_2 = Unet_Down(256, 512)
        self.down2_3 = Unet_Down(512, 1024)

        self.up3_0 = Unet_Up(1024, 512)
        self.up2_1 = Unet_Up(512, 256)
        self.up1_2 = Unet_Up(256, 128)

        self.conv2_1 = SLA(512 * 2, 512)
        self.conv1_2 = SLA(256 * 2, 256)
        self.conv0_3 = SLA(128 * 2, 128)


    def forward(self, input):
        x0_0 = self.conv0_0(input)   # torch.Size([bs, dim, 5, 5])

        # up sampling
        x1_0 = self.down0_1(x0_0)
        x2_0 = self.down1_2(x1_0)
        x3_0 = self.down2_3(x2_0)   # torch.Size([bs, 1024, 1, 1])

        x2_1 = self.conv2_1(x2_0, self.up3_0(x3_0))
        x1_2 = self.conv1_2(x1_0, self.up2_1(x2_1))
        x0_3 = self.conv0_3(x0_0,self.up1_2(x1_2))

        # the concatenated multiscale (CMS)
        f_out5 = x2_1 + x2_0  # torch.Size([bs, 512, 5, 5])
        f_out7 = x1_2 + x1_0  # torch.Size([bs, 256, 7, 7])
        f_out9 = x0_3 + x0_0  # torch.Size([bs, 128, 9, 9])

        return f_out5, f_out7, f_out9

'''
二。 temporal feature encoder–decoder (TFED) subnetworks
'''

def convD(in_ch, out_ch, kernel_size):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )

def convT(in_ch, out_ch, kernel_size):
    return nn.Sequential(
        nn.ConvTranspose2d(in_ch, out_ch, kernel_size),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class MSAF(nn.Module):
    '''
         multi-scale attention fusion (MSAF) module
    '''
    def __init__(self):
        super().__init__()
        self.up35 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, 3),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.down39 = nn.Sequential(
            nn.Conv2d(256, 256, 3),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Conv2d(256, 128, 1)
        self.fc2 = nn.Conv2d(128, 256, 1)
        self.fc3 = nn.Conv2d(128, 256, 1)
        self.fc4 = nn.Conv2d(128, 256, 1)
        self.bn = nn.BatchNorm2d(128)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, d9, d7, d5):
        d7f9 = self.down39(d9)
        d7f5 = self.up35(d5)
        dc = d7f9 + d7f5 + d7
        gp = self.gap(dc)

        gp_ = self.fc1(gp)
        an_ = self.relu(self.bn(gp_))

        gp1 = self.softmax(self.fc2(an_))
        gp2 = self.softmax(self.fc3(an_))
        gp3 = self.softmax(self.fc4(an_))

        atten_img = torch.cat((torch.mul(d7f9, gp1) + d7f9, torch.mul(d7f5, gp2) + d7f5, torch.mul(d7, gp3) + d7),dim=1)

        return atten_img


class MSDFFN(nn.Module):
    def __init__(self,in_ch):
        super(MSDFFN, self).__init__()
        # 1.TFED
        self.fe = NestedUNet(in_ch)
        # 2.gain diffmap
        # Post-processing of differential operation: Channel unification 256
        self.conv0 = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1),
            nn.BatchNorm2d(256),
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1),
            nn.BatchNorm2d(256),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1),
            nn.BatchNorm2d(256),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1),
            nn.BatchNorm2d(256),
        )
        # 3.BDFR module
        self.up35 = convT(256, 256, 3)
        self.up37 = convT(256, 256, 3)
        self.down39 = convD(256, 256, 3)
        self.down37 = convD(256, 256, 3)
        self.convfuse = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1),
            nn.BatchNorm2d(256),
        )

        # 4.MSAF module
        self.msaf = MSAF()

        # 5.output
        self.down71 = nn.Sequential(
            nn.Conv2d(768, 256, kernel_size=3,stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Sequential(
            nn.Linear(3 * 3 * 256, 256, bias=True),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2, bias=True),
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, img_before, img_after):
        # 1.TFED subnetworks
        f1_out5, f1_out7, f1_out9 = self.fe(img_before)
        f2_out5, f2_out7, f2_out9 = self.fe(img_after)

        # 2.gain diffmap
        d_map5 = self.conv1(f2_out5 - f1_out5)
        d_map7 = self.conv2(f2_out7 - f1_out7)
        d_map9 = self.conv3(f2_out9 - f1_out9)

        # 3.BDFR module
        d7f9 = self.down39(d_map9) + d_map7
        d5f7 = self.down37(d7f9) + d_map5
        d7f5 = self.up35(d_map5) + d_map7
        d9f7 = self.up37(d7f5) + d_map9

        input71 = d9f7
        input72 = self.convfuse(torch.cat((d7f9, d7f5), dim=1))
        input73 = d5f7

        # 4.MSAF module
        atten_map = self.msaf(input71, input72, input73)  # d9, d7, d5
        fe_out = atten_map.clone()
        out = self.down71(fe_out)

        # 5.output
        out1 = torch.flatten(out, 1, 3)
        out2 = self.fc(out1)
        final_out = self.softmax(out2)

        return final_out


