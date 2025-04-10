import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """ 殘差塊，包含兩個卷積層和一個跳躍連接 """
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x  # 保存輸入（跳躍連接）

        out = self.conv1(x)
        out = F.relu(self.bn1(out), inplace=True)
        out = self.conv2(out)
        out = self.bn2(out)

        out += identity  # 將跳躍連接的結果添加到最後的輸出
        out = F.relu(out, inplace=True)  # 最終激活

        return out

class EnhancedSRCNN(nn.Module):
    def __init__(self):
        super(EnhancedSRCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, stride=(1, 1), padding=(4, 4))
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=(1, 1), padding=(1, 1))
        self.relu2 = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm2d(64)
        
        # 殘差塊
        self.residual_block = ResidualBlock(64)

        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, stride=(1, 1), padding=(1, 1))
        self.relu3 = nn.ReLU(inplace=True)
        self.bn3 = nn.BatchNorm2d(32)
        
        self.conv4 = nn.Conv2d(32, 3, kernel_size=5, stride=(1, 1), padding=(2, 2))

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.bn2(self.relu2(self.conv2(x)))
        x = self.residual_block(x)
        x = self.bn3(self.relu3(self.conv3(x)))
        x = self.conv4(x)
        return x


if __name__ == '__main__':
    model = EnhancedSRCNN()
    # Total parameters and trainable parameters.
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")