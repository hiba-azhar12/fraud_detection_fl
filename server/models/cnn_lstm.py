import torch
import torch.nn as nn


class FraudCNNLSTM(nn.Module):
    """
    CNN 1D + LSTM — Architecture hybride avancée
    ==============================================
    Amélioration par rapport à l'article FFD (notre contribution) :
    - CNN extrait les patterns locaux entre features voisines
    - LSTM capture les dépendances séquentielles à plus longue portée
    - Combinaison CNN+LSTM dépasse le CNN seul sur données temporelles

    Justification pour le rapport :
    L'article FFD utilise un CNN simple. Nous proposons CNN+LSTM comme
    architecture avancée pour capturer à la fois les patterns locaux
    (types de carte, montants) et les séquences temporelles de transactions.

    IMPORTANT : pas de Sigmoid() final — on utilise BCEWithLogitsLoss.
    """
    def __init__(self, input_dim: int = 37):
        super().__init__()

        # CNN : détecte les patterns locaux entre features voisines
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        # LSTM : capture les dépendances séquentielles
        self.lstm = nn.LSTM(
            input_size=32,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )

        # Classifier final
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            # Pas de Sigmoid() — BCEWithLogitsLoss l'inclut
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)               # (batch, 1, features)
        x = self.conv(x)                 # (batch, 32, features/2)
        x = x.permute(0, 2, 1)          # (batch, features/2, 32) pour LSTM
        _, (hidden, _) = self.lstm(x)
        x = hidden[-1]                   # dernière couche LSTM (batch, 64)
        return self.classifier(x)        # (batch, 1)
