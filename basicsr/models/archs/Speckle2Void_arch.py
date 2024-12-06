import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class Speckle2Void(nn.Module):
    def __init__(self, c=1, L_noise=1):
        super(Speckle2Void, self).__init__()
        self.channels = c
        self.L_noise = L_noise
        self.F = 64
        
        self.conv_layers_y = self._make_conv_layers()
        self.conv_layers_x = self._make_conv_layers()
        self.fusion_layers = self._make_fusion_layers()
        
    #     self._init_weight()

    # def _init_weight(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             nn.init.kaiming_normal_(m.weight)
    #         elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
    #             nn.init.constant_(m.weight, 1)
    #             nn.init.constant_(m.bias,0)

    def _make_conv_layers(self):
        layers = nn.ModuleList()
        _layers = []
        _layers.append(nn.Conv2d(self.channels, self.F, kernel_size=3, padding=1))
        _layers.append(nn.LeakyReLU())
        layers.append(nn.Sequential(*_layers))
        for _ in range(15):
            _layers = []
            _layers.append(nn.Conv2d(self.F, self.F, kernel_size=3, padding=1))
            _layers.append(nn.BatchNorm2d(self.F))
            _layers.append(nn.LeakyReLU())
            layers.append(nn.Sequential(*_layers))
        layers.append(nn.Conv2d(self.F, self.F, kernel_size=3, padding=1))
        return layers

    def _make_fusion_layers(self):
        layers = []
        layers.append(nn.Conv2d(4 * self.F, self.F, kernel_size=1, padding='same'))
        layers.append(nn.LeakyReLU())
        layers.append(nn.Conv2d(self.F, self.F, kernel_size=1, padding='same'))
        layers.append(nn.LeakyReLU())
        layers.append(nn.Conv2d(self.F, 2, kernel_size=1, padding='same'))
        layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    def forward(self, x):
        intermediates = []
        for j in range(4):
            out = torch.rot90(x, k=j, dims=[2, 3])
            if j in [0, 2]:
                for layer in self.conv_layers_y:
                    out = F.pad(out, (0, 0, 1, 0), mode='constant')
                    out = layer(out)
                    out = out[:, :, :-1, :]
                if self.training:
                    shift = random.choice([1, 2])
                    out = self.dynamic_shift(out, shift)
                else:
                    out = self.dynamic_shift(out, 1)
                out = torch.rot90(out, k=4-j, dims=[2, 3])
            else:
                for layer in self.conv_layers_x:
                    out = F.pad(out, (0, 0, 1, 0), mode='constant')
                    out = layer(out)
                    out = out[:, :, :-1, :]
                out = self.dynamic_shift(out, 1)
                out = torch.rot90(out, k=4-j, dims=[2, 3])
            intermediates.append(out)
        
        x = torch.concat(intermediates, dim=1)
        x = self.fusion_layers(x)
        return x

    def dynamic_shift(self, inp, pad_size):
        x1 = F.pad(inp, (0, 0, pad_size, 0), mode='constant')
        return x1[:, :, :-pad_size, :]


if __name__ == "__main__":
    model = Speckle2Void()
    x = torch.randn(1, 1, 256, 256)  # Example input
    out = model(x)
    print(out.shape)  # Should be (1, 1, 256, 256)
