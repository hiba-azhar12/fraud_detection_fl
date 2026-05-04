# FL Finance Platform — Step 3
## Federated Learning pour la Détection de Fraude Bancaire

**Référence scientifique** : Yang et al., *"FFD: A Federated Learning Based Method for Credit Card Fraud Detection"*, BigData 2019.

---

## Architecture

```
fl-finance-platform/
├── client/
│   ├── client.py          ← Client FL (Deep Learning + DP)
│   ├── privacy.py         ← Differential Privacy (Clip + Gauss)
│   ├── Dockerfile
│   └── models/
│       ├── __init__.py    ← Factory get_model()
│       ├── mlp.py         ← MLP baseline
│       ├── cnn1d.py       ← CNN 1D (identique article FFD)
│       └── cnn_lstm.py    ← CNN+LSTM (notre amélioration)
├── server/
│   ├── server.py          ← Serveur FL + FedProx + mTLS
│   └── Dockerfile
├── certs/
│   └── generate_certs.sh  ← Certificats mTLS X.509
├── data/                  ← Parquets des silos (à placer ici)
├── results/               ← JSON résultats par round
├── docker-compose-cnn1d.yml    ← Binôme 1
├── docker-compose-mlp.yml      ← Binôme 2
└── docker-compose-cnnlstm.yml  ← Binôme 3
```

---

## Lancement

### Étape 1 — Générer les certificats mTLS
```bash
cd certs && bash generate_certs.sh && cd ..
```

### Étape 2 — Placer les données
```
data/
├── train_A.parquet   ← Silo Bank A (généré par notebook S1/S2)
├── test_A.parquet
├── train_B.parquet
├── test_B.parquet
├── train_C.parquet
├── test_C.parquet
├── train_D.parquet
├── test_D.parquet
└── test_global.parquet  ← Test global pour évaluation serveur
```

### Étape 3 — Lancer le modèle CNN1D (référence article)
```bash
docker compose -f docker-compose-cnn1d.yml up --build
```

### Autres modèles
```bash
docker compose -f docker-compose-mlp.yml up --build       # MLP baseline
docker compose -f docker-compose-cnnlstm.yml up --build   # CNN+LSTM avancé
```

---

## Modèles Deep Learning

| Modèle | Description | AUC cible |
|--------|-------------|-----------|
| `cnn1d` | CNN identique article FFD — référence | ≥ 0.955 |
| `mlp` | MLP baseline avec BatchNorm | ≥ 0.90 |
| `cnnlstm` | CNN+LSTM hybride — notre amélioration | ≥ 0.96 |

---

## Paramètres alignés avec l'article FFD

| Paramètre | Article FFD | Notre implémentation |
|-----------|-------------|----------------------|
| Local epochs | E=10 (optimal Tableau 4) | 10 |
| Batch size | B=80 (optimal Figure 5) | 80 |
| Learning rate | η=0.01 | 0.01 |
| SMOTE ratio | 1:100 (Section 4.3) | 1:100 (notebook S1/S2) |
| Métriques | F1, AUC, Precision, Recall | F1, AUC, Precision, Recall |
| Rounds | 50 | 20 |

---

## Améliorations par rapport à l'article

| Aspect | Article FFD | Notre projet |
|--------|-------------|-------------|
| Algorithme FL | FedAvg simple | **FedProx** (proximal_mu=0.1) — meilleur sur Non-IID |
| Sécurité | Aucune | **mTLS X.509** Zero Trust |
| Privacy | Aucune | **Differential Privacy** (Clip + Gauss, ε≈3.0) |
| Partition | Shuffle aléatoire | **Dirichlet α=0.5** — simule hétérogénéité réelle |
| Pondération | FedAvg simple | **Équation 7** — nc/n × alpha_c |
| Modèles | CNN seul | MLP + CNN + **CNN+LSTM** hybride |

---

## Résultats attendus (référence article)

```
Round 20 — Global pondéré (Eq.7)
  F1  : ≥ 0.93  (article : 0.9393 à F=0.1)
  AUC : ≥ 0.95  (article : 0.9555 à F=0.1)
```

---

## Sécurité mTLS

Chaque banque possède un certificat X.509 unique signé par la CA.
Un container sans certificat valide est automatiquement rejeté.

```bash
# Test de sécurité — doit être rejeté
docker run --network fl-finance-platform_fl-network \
  fl-finance-platform-bank-a \
  # sans certificat → SSL handshake failed ✗
```
