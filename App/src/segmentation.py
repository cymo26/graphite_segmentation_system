import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np
from PIL import Image

# DEFINICJE ARCHITEKTUR MODELI

# U-Net
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(2)
    
    def forward(self, x):
        skip = self.conv(x)
        pooled = self.pool(skip)
        return skip, pooled


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        self.conv = DoubleConv(out_channels * 2, out_channels)
    
    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

# U-Net binarny
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        self.enc1 = EncoderBlock(in_channels, 64)
        self.enc2 = EncoderBlock(64, 128)
        self.enc3 = EncoderBlock(128, 256)
        self.enc4 = EncoderBlock(256, 512)
        self.bottleneck = DoubleConv(512, 1024)
        self.dec4 = DecoderBlock(1024, 512)
        self.dec3 = DecoderBlock(512, 256)
        self.dec2 = DecoderBlock(256, 128)
        self.dec1 = DecoderBlock(128, 64)
        self.out_conv = nn.Conv2d(64, out_channels, 1)
    
    def forward(self, x):
        s1, p1 = self.enc1(x)
        s2, p2 = self.enc2(p1)
        s3, p3 = self.enc3(p2)
        s4, p4 = self.enc4(p3)
        b = self.bottleneck(p4)
        d4 = self.dec4(b, s4)
        d3 = self.dec3(d4, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)
        return self.out_conv(d1)

# DeepLabV3+
class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels=256):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=18, dilation=18, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        size = x.shape[2:]
        feat1 = self.conv1(x)
        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        feat5 = F.interpolate(self.pool(x), size=size, mode='bilinear', align_corners=False)
        return self.project(torch.cat([feat1, feat2, feat3, feat4, feat5], dim=1))

# DeepLabV3+ z ResNet50
class DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes=1, pretrained=False):
        super().__init__()
        resnet = models.resnet50(weights='IMAGENET1K_V1' if pretrained else None)
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.aspp = ASPP(2048, 256)
        self.low_level_proj = nn.Sequential(
            nn.Conv2d(256, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(256 + 48, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(256, num_classes, 1)
        )

    def forward(self, x):
        size = x.shape[2:]
        x = self.layer0(x)
        low_level = self.layer1(x)
        x = self.layer2(low_level)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.aspp(x)
        x = F.interpolate(x, size=low_level.shape[2:], mode='bilinear', align_corners=False)
        low_level = self.low_level_proj(low_level)
        x = torch.cat([x, low_level], dim=1)
        x = self.decoder(x)
        x = F.interpolate(x, size=size, mode='bilinear', align_corners=False)
        return x


# NORMALIZACJA
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


# FUNKCJE PREDYKCJI

def predict_full_image(model, img_np, device, tile_size=256, overlap=32, 
                       use_imagenet_norm=False, batch_size=8, progress_callback=None):
    h, w = img_np.shape[:2]
    
    # Bufor
    pred_sum = np.zeros((h, w), dtype=np.float32)
    count = np.zeros((h, w), dtype=np.float32)
    
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
    
    if use_imagenet_norm:
        img_tensor = (img_tensor - IMAGENET_MEAN) / IMAGENET_STD
    
    step = tile_size - overlap
    
    # pozycje tile'ow
    tile_positions = []
    for y in range(0, h, step):
        for x in range(0, w, step):
            y_end = min(y + tile_size, h)
            x_end = min(x + tile_size, w)
            y_start = max(0, y_end - tile_size)
            x_start = max(0, x_end - tile_size)
            tile_positions.append((y_start, y_end, x_start, x_end))
    
    total_tiles = len(tile_positions)
    
    # Padding 
    if use_imagenet_norm:
        pad_value = torch.tensor([
            -IMAGENET_MEAN[0] / IMAGENET_STD[0],
            -IMAGENET_MEAN[1] / IMAGENET_STD[1],
            -IMAGENET_MEAN[2] / IMAGENET_STD[2]
        ]).view(3, 1, 1)
    else:
        pad_value = torch.zeros(3, 1, 1)
    
    model.eval()
    use_fp16 = device.type == 'cuda'
    processed = 0
    
    with torch.no_grad():
        # Przetwarzaj w batchach
        for batch_start in range(0, len(tile_positions), batch_size):
            batch_positions = tile_positions[batch_start:batch_start + batch_size]
            tiles = []
            
            for y_start, y_end, x_start, x_end in batch_positions:
                tile = img_tensor[:, y_start:y_end, x_start:x_end]
                
                if tile.shape[1] < tile_size or tile.shape[2] < tile_size:
                    padded = pad_value.expand(3, tile_size, tile_size).clone()
                    padded[:, :tile.shape[1], :tile.shape[2]] = tile
                    tile = padded
                
                tiles.append(tile)
            
            batch_tensor = torch.stack(tiles).to(device)
            
            if use_fp16:
                with torch.amp.autocast('cuda'):
                    logits = model(batch_tensor)
                    preds = torch.sigmoid(logits).float()
            else:
                logits = model(batch_tensor)
                preds = torch.sigmoid(logits)
            
            preds = preds.squeeze(1).cpu().numpy()
            
            # Zapisz wyniki
            for i, (y_start, y_end, x_start, x_end) in enumerate(batch_positions):
                actual_h = y_end - y_start
                actual_w = x_end - x_start
                pred_sum[y_start:y_end, x_start:x_end] += preds[i, :actual_h, :actual_w]
                count[y_start:y_end, x_start:x_end] += 1
            
            # Raportuj postep
            processed += len(batch_positions)
            if progress_callback:
                progress_callback(processed, total_tiles)
    
    pred_avg = pred_sum / np.maximum(count, 1)
    return pred_avg


# Konfiguracja modeli (eksportowana do GUI)
MODELS_CONFIG = {
    'U-Net (bez filtracji)': {
        'folder': 'U-NET_WITHOUT_FILTRATION',
        'file': 'best_unet_pytorch.pth',
        'class': UNet,
        'tile_size': 256,
        'use_imagenet_norm': False
    },
    'U-Net (z filtracja)': {
        'folder': 'U-NET_WITH_FILTRATION',
        'file': 'best_unet_pytorch.pth',
        'class': UNet,
        'tile_size': 256,
        'use_imagenet_norm': False
    },
    'DeepLabV3+': {
        'folder': 'DEEPLABV3_WITH_FILTRATION',
        'file': 'best_deeplabv3_pytorch.pth',
        'class': DeepLabV3Plus,
        'tile_size': 512,
        'use_imagenet_norm': True
    }
}
