import torch
import torch.nn as nn
import torch.nn.functional as F

class SARUSE(nn.Module):
    def __init__(self):
        super(SARUSE, self).__init__()

        # Contracting path (Encoder)
        self.enc_conv1 = self.conv_block(1, 32)
        self.enc_conv2 = self.conv_block(32, 64)
        self.enc_conv3 = self.conv_block(64, 128)
        self.enc_conv4 = self.conv_block(128, 256)

        # Maxpool layers
        self.pool = nn.MaxPool2d(2)

        # Expansive path (Decoder)
        self.upconv3 = self.upconv_block(256, 128)
        self.dec_conv3 = self.conv_block(256, 128)  # 128 from upsampling + 128 from skip connection
        
        self.upconv2 = self.upconv_block(128, 64)
        self.dec_conv2 = self.conv_block(128, 64)  # 64 from upsampling + 64 from skip connection
        
        self.upconv1 = self.upconv_block(64, 32)
        self.dec_conv1 = self.conv_block(64, 32)  # 32 from upsampling + 32 from skip connection

        # Final convolution to reduce channels to 1 (output)
        self.final_conv = nn.Conv2d(32, 1, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def upconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        )

    def forward(self, x):
        # Encoder
        enc1 = self.enc_conv1(x)     # (B, 32, H, W)
        enc2 = self.enc_conv2(self.pool(enc1))  # (B, 64, H/2, W/2)
        enc3 = self.enc_conv3(self.pool(enc2))  # (B, 128, H/4, W/4)
        enc4 = self.enc_conv4(self.pool(enc3))  # (B, 256, H/8, W/8)
        
        # Decoder
        dec3 = self.upconv3(enc4)   # (B, 128, H/4, W/4)
        dec3 = torch.cat((dec3, enc3), dim=1)  # Concatenate with enc3
        dec3 = self.dec_conv3(dec3)  # (B, 128, H/4, W/4)
        
        dec2 = self.upconv2(dec3)   # (B, 64, H/2, W/2)
        dec2 = torch.cat((dec2, enc2), dim=1)  # Concatenate with enc2
        dec2 = self.dec_conv2(dec2)  # (B, 64, H/2, W/2)
        
        dec1 = self.upconv1(dec2)   # (B, 32, H, W)
        dec1 = torch.cat((dec1, enc1), dim=1)  # Concatenate with enc1
        dec1 = self.dec_conv1(dec1)  # (B, 32, H, W)

        # Final output layer
        output = self.final_conv(dec1)  # (B, 1, H, W)
        return output

if __name__ == "__main__":
    model = SARUSE()
    x = torch.randn(1, 1, 256, 256)  # Example input
    out = model(x)
    print(out.shape)  # Should be (1, 1, 256, 256)
