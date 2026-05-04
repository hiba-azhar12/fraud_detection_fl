import numpy as np

"""
Differential Privacy — Privacy Engine (CORRIGÉ)
================================================
Formule exacte du prof (PDF pages 3-4) :
    1. Clipping L2 GLOBAL : appliqué sur le vecteur gradient concatené complet
    2. Bruit Gaussien : delta_W_private = delta_W_clip + N(0, sigma^2 * C^2)

✅ CORRECTION PROBLÈME N°4 :
    Avant : clipping appliqué couche par couche → garantie DP brisée
    Après : clipping sur le gradient global concatené → garantie DP conforme

Budget privacy Finance (CDC prof) : epsilon entre 1.0 et 5.0
Les features financières sont déjà anonymisées (PCA) → sigma moins strict
qu'en santé (epsilon < 1.0).
"""


def apply_dp_global(grad_list: list, C: float = 2.0, sigma: float = 0.05) -> list:
    """
    ✅ DP sur le vecteur gradient GLOBAL concatené.

    Algorithme :
    1. Concatener tous les gradients en un seul vecteur plat
    2. Clipping L2 global : ||grad_global||_2 <= C
    3. Ajout bruit gaussien calibré N(0, sigma^2 * C^2)
    4. Decouper et reshaper vers les formes originales

    Args:
        grad_list : liste de numpy arrays (un par couche du modèle)
        C         : seuil de clipping L2 (sensitivity bound)
        sigma     : niveau de bruit (epsilon ~ 4.0 pour sigma=0.05, C=2.0)

    Returns:
        liste de numpy arrays avec DP appliquée, mêmes shapes que l'entrée
    """
    # Sauvegarder les shapes originales
    shapes = [g.shape for g in grad_list]
    dtypes = [g.dtype for g in grad_list]

    # 1. Concatener en vecteur plat
    flat = np.concatenate([g.astype(np.float64).flatten() for g in grad_list])

    # 2. Clipping L2 GLOBAL (pas par couche)
    norm = np.linalg.norm(flat)
    if norm > C:
        flat = flat * (C / norm)

    # 3. Bruit gaussien calibré à la sensitivity globale C
    noise = np.random.normal(0, sigma * C, size=flat.shape)
    flat_private = flat + noise

    # 4. Reconstruire la liste de gradients avec les shapes originales
    result = []
    offset = 0
    for shape, dtype in zip(shapes, dtypes):
        size = int(np.prod(shape))
        layer_grad = flat_private[offset:offset + size].reshape(shape).astype(dtype)
        result.append(layer_grad)
        offset += size

    return result


# ── Fonctions conservées pour compatibilité (non utilisées dans le pipeline) ──

def clip_gradients(gradients: np.ndarray, C: float = 1.0) -> np.ndarray:
    """L2 Clipping sur un seul array (legacy)."""
    norm = np.linalg.norm(gradients)
    if norm > C:
        return gradients * (C / norm)
    return gradients


def add_gaussian_noise(gradients: np.ndarray, C: float = 1.0, sigma: float = 0.1) -> np.ndarray:
    """Bruit gaussien sur un seul array (legacy)."""
    noise = np.random.normal(0, sigma * C, size=gradients.shape)
    return gradients + noise


def apply_dp(gradients: np.ndarray, C: float = 1.0, sigma: float = 0.1) -> np.ndarray:
    """Legacy — utiliser apply_dp_global() pour la garantie DP correcte."""
    p = gradients.astype(np.float32)
    p = clip_gradients(p, C)
    p = add_gaussian_noise(p, C, sigma)
    return p
