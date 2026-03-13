import torch
import torch.nn as nn

class SEBlock(nn.Module):
    """Squeeze-and-Excitation attention module."""
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ResBlock(nn.Module):
    """Custom Residual Block with SE Attention."""
    def __init__(self, in_channels, out_channels, stride=1, activation=nn.ReLU):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act = activation()
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.se = SEBlock(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += residual
        out = self.act(out)
        return out

class MultiTaskRGBDNet(nn.Module):
    def __init__(self, num_classes=10, activation=nn.ReLU):
        super(MultiTaskRGBDNet, self).__init__()
        
        # --- EARLY FUSION STEM ---
        self.rgb_stem = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1, stride=2), nn.BatchNorm2d(32), activation())
        self.depth_stem = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1, stride=2), nn.BatchNorm2d(16), activation())
        
        # --- RESIDUAL ENCODER ---
        self.enc1 = ResBlock(48, 64, stride=1, activation=activation)
        self.enc2 = ResBlock(64, 128, stride=2, activation=activation)
        self.enc3 = ResBlock(128, 256, stride=2, activation=activation)
        self.enc4 = ResBlock(256, 512, stride=2, activation=activation)

        # --- RESIDUAL DECODER (Segmentation) ---
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.dec3 = ResBlock(512, 256, activation=activation)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.dec2 = ResBlock(256, 128, activation=activation)
        
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.dec1 = ResBlock(128, 64, activation=activation) # 64 (up) + 64 (skip1)
        
        self.final_up = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.final_dec = ResBlock(32, 16, activation=activation) 
        self.final_mask = nn.Conv2d(16, 1, kernel_size=3, padding=1)

        # --- CLASSIFICATION HEAD ---
        self.cls_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(512, 256), activation(), nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

        # --- BOUNDING BOX HEAD ---
        self.box_pool = nn.AdaptiveAvgPool2d((5, 5))
        self.bbox_regressor = nn.Sequential(
            nn.Linear(512 * 5 * 5, 512), activation(), nn.Dropout(0.2),
            nn.Linear(512, 128), activation(),
            nn.Linear(128, 4),
            nn.Sigmoid()
        )

    def forward(self, rgb, depth):
        # 1. Stem Fusion
        r = self.rgb_stem(rgb)
        d = self.depth_stem(depth)
        stem_features = torch.cat((r, d), dim=1) 
        
        # 2. Encoder
        skip1 = self.enc1(stem_features)
        skip2 = self.enc2(skip1)      
        skip3 = self.enc3(skip2)   
        bottleneck = self.enc4(skip3)  

        # 3. Decoder
        x = self.up3(bottleneck)
        x = self.dec3(torch.cat((x, skip3), dim=1))
        
        x = self.up2(x)
        x = self.dec2(torch.cat((x, skip2), dim=1))
        
        x = self.up1(x)
        x = self.dec1(torch.cat((x, skip1), dim=1))
        
        # Upsample to 480x640 and map to 1-channel mask
        x = self.final_up(x) 
        x = self.final_dec(x)
        mask_logits = self.final_mask(x)

        # 4. Task Heads
        class_logits = self.classifier(self.cls_pool(bottleneck).flatten(1))
        bbox_coords = self.bbox_regressor(self.box_pool(bottleneck).flatten(1))

        return class_logits, bbox_coords, mask_logits