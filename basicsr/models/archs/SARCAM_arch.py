import torch
import torch.nn as nn

import torch.nn.functional as F

class _conv_(nn.Module) :
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, bias) :
        # Inheritance
        super(_conv_, self).__init__()

        # Create Layer Instance
        self._conv_ = nn.Conv2d(
                            in_channels = in_channels,
                            out_channels = out_channels,
                            kernel_size = kernel_size,
                            stride = stride,
                            padding = (dilation * (kernel_size - 1)) // 2 ,
                            dilation = dilation,
                            bias = bias
                            )

    def forward(self, x) :
        out = self._conv_(x)

        return out

class _conv_block_(nn.Module) :
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, bias) :
        # Inheritance
        super(_conv_block_, self).__init__()

        # Create Layer Instance
        self._conv_in_ =  _conv_(in_channels, out_channels, kernel_size, stride, dilation, bias)

    def forward(self, x) :
        out = self._conv_in_(x)
        out = F.leaky_relu(out, 0.2, True)

        return  out

class _context_block_(nn.Module) :
    def __init__(self, in_channels, kernel_size, stride, dilation, bias) :
        # Inheritance
        super(_context_block_, self).__init__()

        # Create Layer Instance
        self._conv_in_ = _conv_(in_channels, in_channels, kernel_size, stride, dilation, bias)
        self._d_1_ = _residual_channel_attention_block_(in_channels, kernel_size, stride, dilation, bias)
        self._d_2_ = _residual_channel_attention_block_(in_channels, kernel_size, stride, dilation * 2, bias)
        self._d_3_ = _residual_channel_attention_block_(in_channels, kernel_size, stride, dilation * 3, bias)
        self._d_4_ = _residual_channel_attention_block_(in_channels, kernel_size, stride, dilation * 4, bias)
        self._bottleneck_ = _conv_(in_channels * 4, in_channels, 1, stride, dilation, bias)

    def forward(self, x) :
        out = self._conv_in_(x)
        out = torch.cat([self._d_1_(out), self._d_2_(out), self._d_3_(out), self._d_4_(out)], dim = 1)
        out = self._bottleneck_(out)
        out = out + x

        return out

class _channel_attention_module_(nn.Module) :
    def __init__(self, in_channels, stride, dilation, bias) :
        # Inheritance
        super(_channel_attention_module_, self).__init__()

       # Create Layer Instance
        self._aap_ = nn.AdaptiveAvgPool2d(1)
        self._amp_ = nn.AdaptiveMaxPool2d(1)
        self._conv_ = nn.Sequential(
                            _conv_block_(in_channels, in_channels // 4, 1, stride, dilation, bias),
                            _conv_(in_channels // 4, in_channels, 1, stride, dilation, bias)
                            )

    def forward(self, x) :
        out = self._conv_(self._aap_(x)) + self._conv_(self._amp_(x))
        out = F.sigmoid(out)

        return out

class _spatial_attention_module_(nn.Module) :
    def __init__(self, in_channels, stride, dilation, bias) :
        # Inheritance
        super(_spatial_attention_module_, self).__init__()

        # Create Layer Instance
        self._bottleneck_ = _conv_(2, 1, 7, stride, dilation, bias)

    def forward(self, x) :
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self._bottleneck_(out)
        out = F.sigmoid(out)

        return out

class _ResBlock_CBAM_(nn.Module) :
    def __init__(self, in_channels, kernel_size, stride, dilation, bias) :
        # Inheritance
        super(_ResBlock_CBAM_, self).__init__()

        # Create Layer Instance
        self._conv_in_ = _conv_(in_channels, in_channels, kernel_size, stride, dilation, bias)
        self._conv_out_ = _conv_(in_channels, in_channels, kernel_size, stride, dilation, bias)
        self._cam_ = _channel_attention_module_(in_channels, stride, dilation, bias)
        self._sam_ = _spatial_attention_module_(in_channels, stride, dilation, bias)

    def forward(self, x) :
        out = self._conv_in_(x)
        out = out * self._cam_(out)
        out = out * self._sam_(out)
        out = self._conv_out_(out + x)

        return out

class _residual_channel_attention_block_(nn.Module) :
    def __init__(self, in_channels, kernel_size, stride, dilation, bias) :
        # Inheritance
        super(_residual_channel_attention_block_, self).__init__()

        # Create Layer Instance
        self._layer_ = _conv_block_(in_channels, in_channels, kernel_size, stride, dilation, bias)
        self._conv_ = nn.Sequential(
                                    nn.AdaptiveAvgPool2d(1),
                                    _conv_block_(in_channels, in_channels // 4, 1, stride, dilation, bias),
                                    _conv_(in_channels // 4, in_channels, 1, stride, dilation, bias),
                                    )

    def forward(self, x) :
        out = self._layer_(x)
        out = out * F.sigmoid(self._conv_(out))
        out = out + x

        return out

class _residual_group_(nn.Module) :
    def __init__(self, in_channels, kernel_size, stride, dilation, bias) :
        # Inheritance
        super(_residual_group_, self).__init__()

        # Create Layer Instance
        self._cab_1_ = _residual_channel_attention_block_(in_channels, kernel_size, stride, dilation, bias)
        self._cab_2_ = _residual_channel_attention_block_(in_channels, kernel_size, stride, dilation, bias)
        self._cab_3_ = _residual_channel_attention_block_(in_channels, kernel_size, stride, dilation, bias)
        self._conv_out_ = _conv_(in_channels, in_channels, kernel_size, stride, dilation, bias)

    def forward(self, x) :
        out = self._cab_1_(x)
        out = self._cab_2_(out)
        out = self._cab_3_(out)
        out = self._conv_out_(out)
        out = x + out

        return out

class _upsample_(nn.Module) :
    def __init__(self, scale, in_channels, kernel_size, stride, dilation, bias) :
        # Inheritance
        super(_upsample_, self).__init__()

        # Create Layer Instance
        self._up_ = nn.Sequential(
                            nn.PixelShuffle(scale),
                            _conv_block_(in_channels, in_channels, kernel_size, stride, dilation, bias)
                            )
        self._bottleneck_ = _conv_(in_channels * 2, in_channels, 1, stride, dilation, bias)

    def forward(self, x, skip) :
        out = self._up_(x)
        out = torch.cat((out, skip), dim = 1)
        out = self._bottleneck_(out)

        return out


class SAR_CAM(nn.Module) :
    def __init__(self, scale, in_channels, channels, kernel_size, stride, dilation, bias) :
        # Inheritance
        super(SAR_CAM, self).__init__()

        # Create Layer Instance
        self._conv_in_ = _conv_(in_channels, channels, kernel_size, stride, dilation, bias)
        self._conv_out_ = _conv_(channels, in_channels, kernel_size, stride, dilation, bias)
        self._down_ = nn.MaxPool2d(kernel_size = scale, stride = scale)
        self._cb_ = _context_block_(channels, kernel_size, stride, dilation, bias)
        self._rg_1_ = _residual_group_(channels, kernel_size, stride, dilation, bias)
        self._rg_2_ = _residual_group_(channels, kernel_size, stride, dilation, bias)
        self._rg_3_ = _residual_group_(channels, kernel_size, stride, dilation, bias)
        self._conv_ = _conv_(channels, channels, kernel_size, stride, dilation, bias)
        self._rc_ = _ResBlock_CBAM_(channels, kernel_size, stride, dilation, bias)
        self._up_ = _upsample_(scale, channels, kernel_size, stride, dilation, bias)

    def forward(self, x) :
        out = self._conv_in_(x)
        skip_connection = out
        out = self._down_(out)
        out = self._cb_(out)
        out = self._rg_1_(out)
        concat_1 = out
        out = self._rg_2_(out)
        concat_2 = out
        out = self._rg_3_(out)
        concat_3 = out
        out = self._conv_(concat_1 + concat_2 + concat_3)
        concat_4 = self._rc_(out)
        out = torch.cat([concat_1, concat_2, concat_3, concat_4], dim = 1)
        out = self._up_(out, skip_connection)
        out = self._conv_out_(out)
        out = x + out

        return out

    def initialize_weights(self) :
        for m in self.modules() :
            if isinstance(m, nn.Conv2d) :
                # Apply Xavier Uniform Initialization
                torch.nn.init.xavier_uniform_(m.weight.data)

                if m.bias is not None :
                    m.bias.data.zero_()

def Model(scale, in_channels, channels, kernel_size, stride, dilation, bias) :
    return SAR_CAM(scale, in_channels, channels, kernel_size, stride, dilation, bias)


if __name__ == '__main__':

    _net = SAR_CAM(
                    scale = 2,
                    in_channels = 1,
                    channels = 128,
                    kernel_size = 3,
                    stride = 1,
                    dilation = 1,
                    bias = True
                    )
    _net.eval()
    check_inp = torch.randn(3, 1, 128, 128)
    
    _ = _net(check_inp)

    _inp_shape = (1, 256, 256)

    from ptflops import get_model_complexity_info

    flops, params = get_model_complexity_info(_net, _inp_shape, verbose=False, print_per_layer_stat=True)

    print(f"flops={flops}, params={params}")
    # flops=61.5 GMac, params=3.32 M