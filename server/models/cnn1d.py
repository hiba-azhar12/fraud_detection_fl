import torch
import torch.nn as nn


class FraudCNN1D(nn.Module):
    """
    CNN 1D — Architecture identique à l'article FFD (Yang et al., 2019)
    =====================================================================
    Référence scientifique directe : AUC = 95.5%, F1 = 93.9%
    sur le dataset ULB CreditCard avec Federated Learning.

    Architecture exacte (Section 4.3 de l'article) :
    - Conv1 : 32 channels + MaxPool
    - Conv2 : 64 channels + MaxPool
    - FC    : 512 unités + ReLU
    - Output: 1 neurone (classification binaire)

    IMPORTANT : pas de Sigmoid() final — on utilise BCEWithLogitsLoss
    qui inclut le Sigmoid en interne pour plus de stabilité numérique.
    Supprimer le Sigmoid() final est nécessaire pour corriger le bug pos_weight.
    """
    def __init__(self, input_dim: int = 37):
        super().__init__()

        # Couche Conv1 : 32 channels, kernel 3 — détecte patterns locaux
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        # Couche Conv2 : 64 channels, kernel 3 — patterns plus complexes
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        # Taille après 2 MaxPool de kernel 2
        conv_out_dim = (input_dim // 4) * 64

        # Fully Connected 512 unités + ReLU (exactement comme l'article)
        self.classifier = nn.Sequential(
            nn.Linear(conv_out_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            # Pas de Sigmoid() — BCEWithLogitsLoss l'inclut
            nn.Linear(512, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)           # (batch, features) → (batch, 1, features)
        x = self.conv1(x)            # (batch, 32, features/2)
        x = self.conv2(x)            # (batch, 64, features/4)
        x = x.view(x.size(0), -1)   # flatten → (batch, 64 × features/4)
        return self.classifier(x)    # (batch, 1)
