import torch
import torch.nn as nn


class FraudCNNLSTM(nn.Module):
    """
    CNN 1D + LSTM — Architecture hybride améliorée (v2)
    =====================================================
    Corrections apportées vs v1 :
    - LSTM réduit à 1 couche  (2 couches = surparamétré sur données tabulaires)
    - LayerNorm après LSTM     (stabilise la convergence fédérée inter-clients)
    - Dropout retiré du LSTM   (redondant avec le Dropout du classifier)
    - Classifier inchangé      (Linear 64→32→1, sans Sigmoid)

    IMPORTANT : pas de Sigmoid() final — on utilise BCEWithLogitsLoss.
    Le seuil de décision est fixé à 0.5 côté serveur (plus de seuil dynamique).
    """

    def __init__(self, input_dim: int = 37):
        super().__init__()

        # ── CNN : détecte les patterns locaux entre features voisines ──────
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        # ── LSTM : 1 seule couche, pas de dropout ici ──────────────────────
        #   Raison : 2 couches LSTM sur features tabulaires → overfitting local
        #   sévère chez bank_c (petit dataset). 1 couche suffit à capturer
        #   les dépendances entre groupes de features.
        self.lstm = nn.LSTM(
            input_size=32,
            hidden_size=64,
            num_layers=1,       # était 2 → réduit à 1
            batch_first=True,
            # dropout retiré (ne s'applique qu'entre couches, inutile à 1 couche)
        )

        # ── LayerNorm : normalise la sortie LSTM avant le classifier ────────
        #   Critique en FL : chaque client a une distribution différente.
        #   LayerNorm ramène hidden[-1] dans une plage stable avant agrégation.
        self.norm = nn.LayerNorm(64)

        # ── Classifier final ─────────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            # Pas de Sigmoid() — BCEWithLogitsLoss l'inclut
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)              # (batch, 1, features)
        x = self.conv(x)                # (batch, 32, features/2)
        x = x.permute(0, 2, 1)         # (batch, features/2, 32) → séquence pour LSTM
        _, (hidden, _) = self.lstm(x)
        x = hidden[-1]                  # dernière couche LSTM : (batch, 64)
        x = self.norm(x)                # LayerNorm ← nouveau
        return self.classifier(x)       # (batch, 1)