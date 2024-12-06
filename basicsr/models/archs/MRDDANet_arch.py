import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiscaleBlock(nn.Module):
    def __init__(self, in_channels, num_features=32):
        super(MultiscaleBlock, self).__init__()
        self.start = nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1)
        self.conv3x3 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.conv5x5 = nn.Conv2d(num_features, num_features, kernel_size=5, padding=2)
        self.conv7x7 = nn.Conv2d(num_features, num_features, kernel_size=7, padding=3)
        self.end = nn.Conv2d(3*num_features, num_features, kernel_size=3, padding=1)
    
    def forward(self, x):
        out = F.relu(self.start(x))
        out_3x3 = F.relu(self.conv3x3(out))
        out_5x5 = F.relu(self.conv5x5(out))
        out_7x7 = F.relu(self.conv7x7(out))
        out = self.end(torch.cat([out_3x3, out_5x5, out_7x7], dim=1))
        return F.relu(out)

class RDDABlock(nn.Module):
    def __init__(self, in_channels):
        super(RDDABlock, self).__init__()
        self.block1 = nn.Sequential(
            nn.PReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        )
    
    def forward(self, x):
        out = self.block1(x)
        return out + x

class RDDAB(nn.Module):
    def __init__(self, in_channels):
        super(RDDAB, self).__init__()
        self.block1 = RDDABlock(in_channels)
        self.block2 = RDDABlock(in_channels*2)
        self.block3 = RDDABlock(in_channels*5)
        
        self.conv_attn = nn.Conv2d(in_channels*13, in_channels, kernel_size=3, padding=1)
        
        self.cab = ChannelAttention(in_channels)
        self.pab = PixelAttention(in_channels)
        
        self.end = nn.Conv2d(in_channels*2, in_channels, kernel_size=3, padding=1)
        
    def forward(self, x):
        out1 = torch.cat([x, self.block1(x)], dim=1)  # 2
        out2 = torch.cat([x, out1, self.block2(out1)], dim=1)  # 5
        out3 = torch.cat([x, out1, out2, self.block3(out2)], dim=1)  # 13
        
        atten_in = self.conv_attn(out3)
        out = torch.cat([self.cab(atten_in), self.pab(atten_in)], dim=1)
        
        return x + self.end(out)

class RDB(nn.Module):
    def __init__(self, in_channels):
        super(RDB, self).__init__()
        self.rddab = nn.Sequential(
            RDDAB(in_channels),
            RDDAB(in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        )
        
    def forward(self, x):
        out = self.rddab(x)
        return out + x

class ChannelAttention(nn.Module):
    def __init__(self, in_channels):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        y = self.avg_pool(x)
        y = self.fc(y)
        return x * y

class PixelAttention(nn.Module):
    def __init__(self, in_channels):
        super(PixelAttention, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        y = self.conv(F.relu(self.conv1(x)))
        y = self.sigmoid(y)
        return x * y


class MRDDANet(nn.Module):
    def __init__(self, in_channels=1, num_features=32):
        super(MRDDANet, self).__init__()
        self.msb = MultiscaleBlock(in_channels, num_features)
        self.rdb0 = RDB(num_features)
        self.rdb1 = RDB(num_features)
        self.rdb2 = RDB(num_features)
        self.conv_final = nn.Conv2d(num_features*3, in_channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        msb_out = self.msb(x)
        rdb0 = self.rdb0(msb_out)
        rdb1 = self.rdb1(rdb0)
        rdb2 = self.rdb2(rdb1)
        rdb = torch.concat([rdb0, rdb1, rdb2], dim=1)
        rdb = self.conv_final(rdb)
        return rdb + x  # Global residual learning


if __name__ == '__main__':

    _net = MRDDANet(1, 32)
    _net.eval()
    check_inp = torch.randn(3, 1, 128, 128)
    
    _ = _net(check_inp)

    _inp_shape = (1, 256, 256)

    from ptflops import get_model_complexity_info

    flops, params = get_model_complexity_info(_net, _inp_shape, verbose=False, print_per_layer_stat=True)

    print(f"flops={flops}, params={params}")
    # flops=281.98 GMac, params=4.31 M
