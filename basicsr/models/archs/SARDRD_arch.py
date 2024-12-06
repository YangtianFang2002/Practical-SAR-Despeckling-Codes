import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        return x

class NonLocalBlock(nn.Module):
    def __init__(self, in_channels):
        super(NonLocalBlock, self).__init__()
        self.in_channels = in_channels
        self.inter_channels = in_channels // 2
        self.g = nn.Conv2d(in_channels, self.inter_channels, 1)
        self.theta = nn.Conv2d(in_channels, self.inter_channels, 1)
        self.phi = nn.Conv2d(in_channels, self.inter_channels, 1)
        self.W = nn.Conv2d(self.inter_channels, in_channels, 1)
        
    def forward(self, x):
        batch_size, C, H, W = x.size()
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        phi_x = phi_x.permute(0, 2, 1)

        f = torch.matmul(phi_x, theta_x)  # HW x HW
        f_div_C = F.softmax(f, dim=-1)
        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1)
        y = y.view(batch_size, self.inter_channels, H, W)
        W_y = self.W(y)
        z = W_y + x

        return z

class DetailRecoveryBlock(nn.Module):
    def __init__(self, in_channels, dilation_rates):
        super(DetailRecoveryBlock, self).__init__()
        self.convs1 = nn.ModuleList([
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=rate, dilation=rate) 
            for rate in dilation_rates
        ])
        self.convs2 = nn.ModuleList([
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=rate, dilation=rate) 
            for rate in dilation_rates
        ])
        self.conv_bn_relu = ConvBNReLU(in_channels * len(dilation_rates), in_channels, kernel_size=1, padding=0)
        self.conv1 = nn.Conv2d(in_channels * len(dilation_rates), in_channels, kernel_size=1, padding=0)
        self.bn = nn.BatchNorm2d(in_channels)
    
    def forward(self, x):
        out = self.conv_bn_relu(torch.cat([conv(x) for conv in self.convs1], dim=1))
        out = self.bn(self.conv1(torch.cat([conv(out) for conv in self.convs2], dim=1)))
        return out + x


class SAR_DRDNet(nn.Module):
    def __init__(self, num_blocks=5, num_channels=64):
        super(SAR_DRDNet, self).__init__()
        self.initial = ConvBNReLU(1, num_channels)
        
        self.nlb_drbs = nn.ModuleList()
        for _ in range(num_blocks):
            self.nlb_drbs.append(NonLocalBlock(num_channels))
            self.nlb_drbs.append(DetailRecoveryBlock(num_channels, [1, 3, 5]))
        
        self.final = nn.Sequential(
            ConvBNReLU(num_channels, num_channels),
            nn.Conv2d(num_channels, 1, kernel_size=3, padding=1)
        )
        
    def forward(self, x):
        residual = x
        out = self.initial(x)
        
        for layer in self.nlb_drbs:
            out = layer(out)
        
        out = self.final(out)
        out = out + residual
        
        return out



if __name__ == '__main__':

    _net = SAR_DRDNet()
    _net.eval()
    check_inp = torch.randn(3, 1, 256, 256)
    
    _ = _net(check_inp)

    _inp_shape = (1, 256, 256)

    from ptflops import get_model_complexity_info

    flops, params = get_model_complexity_info(_net, _inp_shape, verbose=False, print_per_layer_stat=True)

    print(f"flops={flops}, params={params}")
    # flops=61.5 GMac, params=3.32 M
