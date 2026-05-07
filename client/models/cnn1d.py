import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionModule(nn.Module):
    """
    Self-attention Q/K/V — Eq.4 du papier FFD
    Attention(Q,K,V) = softmax(QKᵀ/√dk) V
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key   = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.scale = hidden_dim ** 0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (batch, channels, seq_len)
        x_t = x.permute(0, 2, 1)                              # (B, seq, channels)
        Q = self.query(x_t)
        K = self.key(x_t)
        V = self.value(x_t)
        scores  = torch.bmm(Q, K.permute(0, 2, 1)) / self.scale  # (B, seq, seq)
        weights = F.softmax(scores, dim=-1)
        out     = torch.bmm(weights, V)                        # (B, seq, channels)
        return out.permute(0, 2, 1)                            # (B, channels, seq)


class FraudCNN1D(nn.Module):
    """CNN 1D simple — référence IJRTI 2023 + FFD Yang 2019"""
    def __init__(self, input_dim: int = 37):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=0), nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, padding=0), nn.ReLU()
        )
        self.pool = nn.MaxPool1d(kernel_size=2)

        after_conv1  = input_dim - 3 + 1   # 35
        after_conv2  = after_conv1 - 3 + 1 # 33
        after_pool   = after_conv2 // 2     # 16
        conv_out_dim = after_pool * 64      # 1024

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_out_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.conv1(x)   # (B, 32, 35)
        x = self.conv2(x)   # (B, 64, 33)
        x = self.pool(x)    # (B, 64, 16)
        return self.classifier(x)


class FraudCNN1DAttention(nn.Module):
    """CNN 1D + Self-Attention Q/K/V — version corrigée"""
    def __init__(self, input_dim: int = 37):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=0), nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, padding=0), nn.ReLU()
        )
        # AttentionModule appliqué sur les 33 timesteps AVANT le pool
        self.attention = AttentionModule(hidden_dim=64)
        self.pool      = nn.AdaptiveAvgPool1d(8)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)      # (B, 1,  37)
        x = self.conv1(x)       # (B, 32, 35)
        x = self.conv2(x)       # (B, 64, 33)
        x = self.attention(x)   # (B, 64, 33) — AttentionModule gère le permute en interne
        x = self.pool(x)        # (B, 64,  8)
        return self.classifier(x)


def get_model(model_type: str, input_dim: int = 37) -> nn.Module:
    if model_type == "cnn1d":
        return FraudCNN1D(input_dim)
    elif model_type == "cnn1d_attention":
        return FraudCNN1DAttention(input_dim)
    else:
        raise ValueError(
            f"Modele inconnu : {model_type}. "
            f"Choisir parmi : cnn1d, cnn1d_attention"
        )