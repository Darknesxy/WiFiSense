"""
WiFiSense - Deep Learning Architectures
PyTorch Convolutional Neural Networks designed for WiFi CSI Human Activity Recognition (HAR).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CSI_1D_CNN(nn.Module):
    """
    1D Convolutional Neural Network for Multi-Channel Temporal CSI Signals.
    Input Shape: (batch_size, n_subcarriers, sequence_length)
    """

    def __init__(self, in_channels: int = 56, num_classes: int = 6, dropout_rate: float = 0.3):
        super(CSI_1D_CNN, self).__init__()

        # Conv Block 1
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=7, stride=1, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        # Conv Block 2
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        # Conv Block 3
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.AdaptiveAvgPool1d(1)

        # Classifier Head
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expected input: (B, C, L) where C = n_subcarriers, L = sequence_length
        # If passed as (B, L, C), transpose it:
        if x.dim() == 3 and x.shape[1] > x.shape[2]:
            x = x.transpose(1, 2)

        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))

        x = x.view(x.size(0), -1)  # Flatten (B, 256)
        x = self.dropout(F.relu(self.fc1(x)))
        out = self.fc2(x)
        return out


class CSI_2D_CNN(nn.Module):
    """
    2D Convolutional Neural Network treating the CSI Time-Frequency matrix as an image.
    Input Shape: (batch_size, 1, sequence_length, n_subcarriers)
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 6, dropout_rate: float = 0.3):
        super(CSI_2D_CNN, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=(5, 3), padding=(2, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2))
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2))
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )

        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expected input: (B, 1, H, W)
        if x.dim() == 3:
            x = x.unsqueeze(1)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.size(0), -1)
        out = self.classifier(x)
        return out


class CSI_CNN_Attention(nn.Module):
    """
    1D-CNN with Temporal Self-Attention for prioritizing dynamic burst events (e.g. Falls).
    """

    def __init__(self, in_channels: int = 56, num_classes: int = 6):
        super(CSI_CNN_Attention, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        # Attention layer over reduced temporal steps
        self.attention_weights = nn.Sequential(
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3 and x.shape[1] > x.shape[2]:
            x = x.transpose(1, 2)

        feat = self.feature_extractor(x)  # (B, 128, T')
        feat = feat.transpose(1, 2)       # (B, T', 128)

        attn = self.attention_weights(feat)     # (B, T', 1)
        attn_norm = F.softmax(attn, dim=1)      # (B, T', 1)

        context = torch.sum(feat * attn_norm, dim=1)  # (B, 128)
        out = self.fc(context)
        return out
