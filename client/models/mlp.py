import torch
import torch.nn as nn


class FraudMLP(nn.Module):
    """
    MLP — Multi-Layer Perceptron (baseline)
    =========================================
    Architecture avec LayerNorm (remplace BatchNorm pour la compatibilité FL).

    En Federated Learning, BatchNorm1d pose problème car ses running_mean/var
    sont calculés sur les données locales de chaque client puis moyennés côté
    serveur, ce qui revient à agréger des statistiques de distributions
    hétérogènes. LayerNorm normalise par exemple plutôt que par batch —
    ses paramètres (gamma, beta) sont des scalaires apprenables qui s'agrègent
    correctement via FedAvg.

    IMPORTANT : pas de Sigmoid() final — on utilise BCEWithLogitsLoss
    qui inclut le Sigmoid en interne pour plus de stabilité numérique.
    """
    def __init__(self, input_dim: int = 37):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 32),
            nn.ReLU(),

            # Pas de Sigmoid() — BCEWithLogitsLoss l'inclut
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
