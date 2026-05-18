import numpy as np
import json
import os
from datetime import datetime


class BehavioralAnalyzer:

    def __init__(self, window: int = 5, mad_threshold: float = 4.5,
                 trust_min: float = 0.1, min_drift_pct: float = 0.05,
                 z_thresh: float = 3.5):
        self.window         = window
        self.mad_threshold  = mad_threshold
        self.trust_min      = trust_min
        self.min_drift_pct  = min_drift_pct
        self.z_thresh       = z_thresh

        self.alpha_history  = {}
        self.trust_scores   = {}
        self.alert_log      = []
        self.round_reports  = []
        self.metric_history = {}

    def update_metrics(self, bank_id: str, f1: float, auc: float, alpha: float) -> None:
        if bank_id not in self.metric_history:
            self.metric_history[bank_id] = []
        self.metric_history[bank_id].append({"f1": f1, "auc": auc, "alpha": alpha})

    def analyze_behavior(self, bank_id: str) -> float:
        hist = self.metric_history.get(bank_id, [])
        if len(hist) < 5:
            return 1.0

        recent   = hist[-self.window:]
        f1s      = [h["f1"]  for h in recent]
        aucs     = [h["auc"] for h in recent]
        penalty  = 1.0
        last_f1  = f1s[-1]
        mean_f1  = np.mean(f1s[:-1]) if len(f1s) > 1 else f1s[0]
        std_f1   = np.std(f1s[:-1])  if len(f1s) > 1 else 0.0
        mean_auc = np.mean(aucs)

        if std_f1 > 0 and (mean_f1 - last_f1) > self.z_thresh * std_f1 and (mean_f1 - last_f1) > 0.15:
            drop    = (mean_f1 - last_f1) / max(mean_f1, 1e-6)
            penalty *= max(0.5, 1.0 - drop)
            self._alert(bank_id, "F1_DROP",
                        f"chute F1 ({mean_f1:.4f} -> {last_f1:.4f}), penalite={penalty:.2f}")

        if mean_auc < 0.70:
            penalty *= 0.5
            self._alert(bank_id, "AUC_FAIBLE",
                        f"AUC moyenne={mean_auc:.4f} < 0.70, penalite={penalty:.2f}")

        if last_f1 >= 0.999:
            penalty *= 0.7
            self._alert(bank_id, "F1_SUSPECT",
                        f"F1=1.0 suspect (possible leakage), penalite={penalty:.2f}")

        return float(np.clip(penalty, 0.5, 1.0))

    def detect_free_rider(self, bank_id: str, params: list,
                          threshold: float = 1e-2) -> bool:
        norm  = self._gradient_norm(params)
        is_fr = norm < threshold
        if is_fr:
            self._alert(bank_id, "FREE_RIDER",
                        f"norm L2 = {norm:.2e} < seuil {threshold:.2e}")
        return is_fr

    def detect_poisoning_mad(self, all_grads: dict,
                             num_samples: dict = None) -> dict:
        if len(all_grads) < 2:
            return {k: False for k in all_grads}

        bank_ids = sorted(all_grads.keys())

        norms = []
        for b in bank_ids:
            norm = self._gradient_norm(all_grads[b])
            if num_samples and b in num_samples and num_samples[b] > 0:
                norm = norm / np.sqrt(num_samples[b])
            norms.append(norm)
        norms = np.array(norms)

        median = np.median(norms)
        mad    = np.median(np.abs(norms - median))

        results = {}
        for bank_id, norm_val in zip(bank_ids, norms):
            if mad < 1e-10:
                is_anomaly = False
                mad_score  = 0.0
            else:
                mad_score  = 0.6745 * abs(norm_val - median) / mad
                is_anomaly = bool(mad_score > self.mad_threshold)

            results[bank_id] = is_anomaly
            if is_anomaly:
                self._alert(
                    bank_id, "POISONING_SUSPECT",
                    f"MAD score = {mad_score:.3f} > seuil {self.mad_threshold} "
                    f"(norm_norm={norm_val:.4f}, mediane={median:.4f}, MAD={mad:.4f})"
                )
            else:
                print(f"[Behavioral] {bank_id} OK — "
                      f"norm_norm={norm_val:.4f}  MAD_score={mad_score:.3f}")

        return results

    def detect_poisoning_isolation_forest(self, all_grads: dict,
                                          num_samples: dict = None) -> dict:
        return self.detect_poisoning_mad(all_grads, num_samples)

    def detect_alpha_drift(self, bank_id: str, alpha: float) -> bool:
        if bank_id not in self.alpha_history:
            self.alpha_history[bank_id] = []
        self.alpha_history[bank_id].append(alpha)

        history = self.alpha_history[bank_id]
        if len(history) >= self.window:
            recent         = history[-self.window:]
            is_monotone    = all(recent[i] > recent[i + 1]
                                 for i in range(len(recent) - 1))
            total_drop     = (recent[0] - recent[-1]) / max(recent[0], 1e-6)
            is_significant = total_drop >= self.min_drift_pct

            if is_monotone and is_significant:
                self._alert(
                    bank_id, "ALPHA_DRIFT",
                    f"Degradation sur {self.window} rounds : "
                    f"{[round(v, 3) for v in recent]} "
                    f"(chute totale = {total_drop*100:.1f}%)"
                )
                return True
        return False

    def compute_trust_score(self, bank_id: str, alpha: float,
                            is_free_rider: bool, is_poison: bool,
                            alpha_max: float = 5.0,
                            all_alphas: dict = None) -> float:
        prev = self.trust_scores.get(bank_id, 1.0)

        if all_alphas and len(all_alphas) >= 2:
            alpha_values = list(all_alphas.values())
            alpha_min    = min(alpha_values)
            alpha_max_v  = max(alpha_values)
            if alpha_max_v > alpha_min:
                quality = float(np.clip(
                    (alpha - alpha_min) / (alpha_max_v - alpha_min),
                    0.5,
                    1.0
                ))
            else:
                quality = 1.0
        else:
            quality = float(np.clip(alpha / max(alpha_max, 1e-6), 0.0, 1.0))

        behavior_factor = self.analyze_behavior(bank_id)

        penalty   = (0.4 if is_free_rider else 0.0) + (0.2 if is_poison else 0.0)
        new_score = 0.6 * prev + 0.4 * quality * behavior_factor - penalty
        new_score = float(np.clip(new_score, self.trust_min, 1.0))
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
                "alpha_history" : {
                    k: [float(v) for v in hist]
                    for k, hist in self.alpha_history.items()
                },
                "metric_history": {
                    k: hist for k, hist in self.metric_history.items()
                },
                "alerts"        : self.alert_log,
                "round_reports" : self.round_reports,
            }, f, indent=2)
        print(f"[Behavioral] Rapport sauvegarde : {path}")

    def summary(self) -> dict:
        return {k: round(v, 4) for k, v in self.trust_scores.items()}

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