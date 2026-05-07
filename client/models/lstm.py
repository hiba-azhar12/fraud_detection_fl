import torch
import torch.nn as nn


class FraudLSTM(nn.Module):
    """
    Pure LSTM — Federated Fraud Detection (v1)
    ===========================================
    Améliorations vs CNN-LSTM :
    - Projection linéaire de l'input (37 → 64) avant LSTM
      → remplace le rôle du CNN, sans biais d'ordre des features
    - Bidirectional LSTM : capture les dépendances dans les deux sens
      → concat(forward, backward) → hidden_size * 2 = 128
    - LayerNorm après LSTM (critique en FL : distributions hétérogènes)
    - Classifier 128 → 64 → 1 avec GELU + Dropout(0.3)
    - Pas de Sigmoid() final → BCEWithLogitsLoss
    """

    def __init__(self, input_dim: int = 37, hidden_size: int = 64, num_layers: int = 1):
        super().__init__()

        # ── Projection input : remplace le CNN ──────────────────────────────
        #   Le CNN ordonnait les features spatialement (inutile sur tabulaire).
        #   Une projection linéaire appris un embedding dense de chaque feature
        #   sans supposer de voisinage entre elles.
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
        )

        # ── Bidirectional LSTM ───────────────────────────────────────────────
        #   bidirectional=True : le modèle voit la séquence de features
        #   dans les deux sens → meilleure capture des co-dépendances.
        #   La sortie hidden est de taille hidden_size * 2 = 128.
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )

        lstm_out_dim = hidden_size * 2  # 128 (bidirectional)

        # ── LayerNorm post-LSTM ──────────────────────────────────────────────
        self.norm = nn.LayerNorm(lstm_out_dim)

        # ── Classifier ───────────────────────────────────────────────────────
        #   GELU > ReLU pour les features continues (fraude financière).
        #   Dropout(0.3) légèrement plus fort qu'avant : le modèle est plus
        #   expressif (bidirectional), on régularise un peu plus.
        self.classifier = nn.Sequential(
            nn.Linear(lstm_out_dim, 64),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (batch, input_dim)

        x = self.input_proj(x)          # (batch, hidden_size)
        x = x.unsqueeze(1)              # (batch, 1, hidden_size) — séquence de longueur 1
        _, (hidden, _) = self.lstm(x)   # hidden : (2, batch, hidden_size) — bidir

        # Concat forward + backward hidden states
        # hidden[0] = forward, hidden[1] = backward
        x = torch.cat([hidden[0], hidden[1]], dim=-1)  # (batch, hidden_size * 2)

        x = self.norm(x)                # LayerNorm
        return self.classifier(x)       # (batch, 1)