from .mlp import FraudMLP
from .cnn1d import FraudCNN1D, FraudCNN1DAttention, AttentionModule
from .cnn_lstm import FraudCNNLSTM
from .lstm import FraudLSTM


def get_model(model_type: str, input_dim: int = 37):
    """
    Factory — switche entre les 4 architectures Deep Learning.

    Args:
        model_type : 'mlp' | 'cnn1d' | 'cnnlstm' | 'lstm'
        input_dim  : nombre de features (37 après feature engineering)

    Modèles :
        mlp     — MLP baseline avec BatchNorm + Dropout
        cnn1d   — CNN identique article FFD (AUC 95.5% référence)
        cnnlstm — CNN+LSTM hybride (notre amélioration vs article)
        lstm    — Pure LSTM bidirectionnel avec projection input (v1)
    """
    models = {
        'mlp':      FraudMLP(input_dim),
        'cnn1d':    FraudCNN1D(input_dim),
        'cnnlstm':  FraudCNNLSTM(input_dim),
        'cnn1d_attention':  FraudCNN1DAttention(input_dim),
        'lstm':     FraudLSTM(input_dim),
    }
    if model_type not in models:
        raise ValueError(
            f"Modèle inconnu : '{model_type}'. "
            f"Choix valides : {list(models.keys())}"
        )
    return models[model_type]


__all__ = ['FraudMLP', 'FraudCNN1D', 'FraudCNNLSTM', 'FraudLSTM', 'FraudCNN1DAttention' ,'get_model']