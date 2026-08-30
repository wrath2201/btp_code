import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, rate=4):
        super(ChannelAttention, self).__init__()
        reduced_channels = max(1, in_channels // rate)
        self.fc1 = nn.Linear(in_channels, reduced_channels)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(reduced_channels, in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x is (b, c, l)
        b, c, l = x.size()
        # Paper implies Linear applied to channel dim without GAP
        # So we permute to (b, l, c) to apply Linear on c
        x_perm = x.transpose(1, 2)
        attn = self.fc1(x_perm)
        attn = self.relu(attn)
        attn = self.fc2(attn)
        attn = self.sigmoid(attn)
        # attn is (b, l, c) -> permute back to (b, c, l)
        attn = attn.transpose(1, 2)
        return x * attn

class SequentialAttention(nn.Module):
    def __init__(self, in_channels, rate=4):
        super(SequentialAttention, self).__init__()
        reduced_channels = max(1, in_channels // rate)
        self.conv1 = nn.Conv1d(in_channels, reduced_channels, kernel_size=7, stride=1, padding=3)
        self.bn1 = nn.BatchNorm1d(reduced_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(reduced_channels, in_channels, kernel_size=7, stride=1, padding=3)
        self.bn2 = nn.BatchNorm1d(in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, s):
        # s is (b, c, l), output of channel attention
        attn = self.conv1(s)
        attn = self.bn1(attn)
        attn = self.relu(attn)
        attn = self.conv2(attn)
        attn = self.bn2(attn)
        attn = self.sigmoid(attn)
        return s * attn

class MGCBlock(nn.Module):
    def __init__(self, in_channels, out_channels, rate=4):
        super(MGCBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.channel_attn = ChannelAttention(out_channels, rate=rate)
        self.seq_attn = SequentialAttention(out_channels, rate=rate)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.channel_attn(x)
        x = self.seq_attn(x)
        x = self.pool(x)
        return x

class MGCNNSDTransformer(nn.Module):
    def __init__(self, num_classes=29, rate=4, d_model=128, nhead=4, num_layers=2, dim_feedforward=512, dropout=0.1):
        super(MGCNNSDTransformer, self).__init__()
        
        # MGCNN Feature Extractor
        self.mgc1 = MGCBlock(in_channels=1, out_channels=16, rate=rate)
        self.mgc2 = MGCBlock(in_channels=16, out_channels=32, rate=rate)
        self.mgc3 = MGCBlock(in_channels=32, out_channels=64, rate=rate)
        self.mgc4 = MGCBlock(in_channels=64, out_channels=128, rate=rate)
        
        # SDTransformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            activation="relu",
            batch_first=True,
            norm_first=True  # Pre-LN as shown in Fig 4 of the paper
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classifier
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x is expected to be (b, 1024) or (b, 1, 1024)
        if x.dim() == 2:
            x = x.unsqueeze(1)
            
        # MGCNN
        x = self.mgc1(x)  # (b, 16, 512)
        x = self.mgc2(x)  # (b, 32, 256)
        x = self.mgc3(x)  # (b, 64, 128)
        x = self.mgc4(x)  # (b, 128, 64)
        
        # SDTransformer expects (batch, seq, feature)
        x = x.transpose(1, 2)  # (b, 64, 128)
        
        # No positional encoding added per the paper
        x = self.transformer(x)
        
        # Classifier
        x = x.transpose(1, 2)  # (b, 128, 64)
        x = self.adaptive_pool(x)  # (b, 128, 1)
        x = x.squeeze(-1)  # (b, 128)
        x = self.fc(x)  # (b, num_classes)
        
        return x
