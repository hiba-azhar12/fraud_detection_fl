import numpy as np
import json
import os
from datetime import datetime
from sklearn.ensemble import IsolationForest


class BehavioralAnalyzer:

    def __init__(self, window: int = 5, contamination: float = 0.1, trust_min: float = 0.3):
        self.window        = window
        self.contamination = contamination
        self.trust_min     = trust_min

        self.alpha_history  = {}
        self.trust_scores   = {}
        self.alert_log      = []
        self.round_reports  = []

    def detect_free_rider(self, bank_id: str, params: list, threshold: float = 1e-2) -> bool:
        norm = self._gradient_norm(params)
        is_fr = norm < threshold
        if is_fr:
            self._alert(bank_id, "FREE_RIDER",
                        f"norm L2 = {norm:.2e} < seuil {threshold:.2e}")
        return is_fr

    def detect_poisoning_isolation_forest(self, all_grads: dict) -> dict:
        if len(all_grads) < 3:
            return {k: False for k in all_grads}

        bank_ids = sorted(all_grads.keys())
        norms    = np.array([[self._gradient_norm(all_grads[b])] for b in bank_ids])

        clf   = IsolationForest(contamination=self.contamination, random_state=42)
        preds = clf.fit_predict(norms)

        results = {}
        for bank_id, pred, norm_val in zip(bank_ids, preds, norms):
            is_anomaly = (pred == -1)
            results[bank_id] = is_anomaly
            if is_anomaly:
                self._alert(bank_id, "POISONING_SUSPECT",
                            f"Isolation Forest — norm L2 = {norm_val[0]:.4f} (outlier)")
        return results

    def detect_alpha_drift(self, bank_id: str, alpha: float) -> bool:
        if bank_id not in self.alpha_history:
            self.alpha_history[bank_id] = []
        self.alpha_history[bank_id].append(alpha)

        history = self.alpha_history[bank_id]
        if len(history) >= self.window:
            recent = history[-self.window:]
            is_drift = all(recent[i] > recent[i + 1] for i in range(len(recent) - 1))
            if is_drift:
                self._alert(bank_id, "ALPHA_DRIFT",
                            f"Degradation sur {self.window} rounds : {[round(v,3) for v in recent]}")
                return True
        return False

    def compute_trust_score(self, bank_id: str, alpha: float, is_free_rider: bool,
                            is_poison: bool, alpha_max: float = 5.0) -> float:
        prev    = self.trust_scores.get(bank_id, 1.0)
        quality = np.clip(alpha / max(alpha_max, 1e-6), 0.0, 1.0)
        penalty = (0.4 if is_free_rider else 0.0) + (0.3 if is_poison else 0.0)

        new_score = 0.7 * prev + 0.3 * quality - penalty
        new_score = float(np.clip(new_score, 0.0, 1.0))
        self.trust_scores[bank_id] = new_score
        return new_score

    def record_round(self, rnd: int, per_bank: dict) -> dict:
        clean_per_bank = {
            bid: {
                "trust"     : float(data["trust"]),
                "is_fr"     : bool(data["is_fr"]),
                "is_poison" : bool(data["is_poison"]),
                "alpha"     : float(data["alpha"]),
            }
            for bid, data in per_bank.items()
        }
        report = {
            "round"       : rnd,
            "timestamp"   : datetime.now().isoformat(),
            "per_bank"    : clean_per_bank,
            "alerts"      : [a for a in self.alert_log if a.get("round") == rnd],
            "trust_scores": {k: float(v) for k, v in self.trust_scores.items()},
        }
        self.round_reports.append(report)
        return report

    def save_report(self, path: str = "/app/results/behavioral_report.json") -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump({
                "trust_scores"  : {k: float(v) for k, v in self.trust_scores.items()},
                "alpha_history" : {k: [float(v) for v in hist] for k, hist in self.alpha_history.items()},
                "alerts"        : self.alert_log,
                "round_reports" : self.round_reports,
            }, f, indent=2)
        print(f"[Behavioral] Rapport sauvegarde : {path}")

    def _gradient_norm(self, params: list) -> float:
        flat = np.concatenate([p.flatten() for p in params])
        return float(np.linalg.norm(flat))

    def _alert(self, bank_id: str, alert_type: str, detail: str) -> None:
        msg = f"[Behavioral] ALERTE {bank_id} — {alert_type} : {detail}"
        print(msg)
        self.alert_log.append({
            "bank_id"   : bank_id,
            "type"      : alert_type,
            "detail"    : detail,
            "timestamp" : datetime.now().isoformat(),
        })
