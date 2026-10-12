import torch
import torch.nn as nn

class EarlyStopping:
    def __init__(self, patience=4, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, stride=1):
        super().__init__()
        padding = kernel_size // 2 if stride == 1 else (kernel_size // 2)
        self.conv = nn.Conv2d(in_c, out_c, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class B1_AlexNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(1, 16, 5),
            nn.MaxPool2d(2, 2),
            ConvBlock(16, 32, 5),
            nn.MaxPool2d(2, 2),
            ConvBlock(32, 64, 3),
            nn.MaxPool2d(2, 2),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64, bias=True),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(64, 32, bias=True),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(32, 1, bias=True)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

class B2_VGG(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(1, 8, 3),
            ConvBlock(8, 8, 3),
            nn.MaxPool2d(2, 2),
            ConvBlock(8, 16, 3),
            ConvBlock(16, 16, 3),
            nn.MaxPool2d(2, 2),
            ConvBlock(16, 32, 3),
            ConvBlock(32, 32, 3),
            nn.MaxPool2d(2, 2),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32, 1, bias=True)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

class ResBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1 = ConvBlock(in_c, out_c, 3, stride=stride)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c)
            )
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return self.relu(out)

class B3_ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.prep = nn.Sequential(
            ConvBlock(1, 16, 3),
            nn.MaxPool2d(2, 2)
        )
        self.layer1 = ResBlock(16, 16, stride=1)
        self.layer2 = ResBlock(16, 32, stride=2)
        self.layer3 = ResBlock(32, 64, stride=2)
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, 1, bias=True)

    def forward(self, x):
        x = self.prep(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)