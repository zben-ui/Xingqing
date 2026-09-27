from __future__ import annotations


class ScreeningClassifier:
    """Deterministic support-priority classification, not a trained clinical model."""

    FEATURE_ORDER = (
        "upi_flag", "scl90_depression_flag", "scl90_anxiety_flag",
        "scl90_interpersonal_sensitivity_flag", "scl90_obsessive_flag",
        "scl90_psychoticism_flag", "scl90_paranoid_flag", "scl90_hostility_flag",
        "suicidal_ideation",
    )
    FACTOR_FLAGS = {
        "depression": "scl90_depression_flag", "anxiety": "scl90_anxiety_flag",
        "interpersonal": "scl90_interpersonal_sensitivity_flag",
        "obsessive": "scl90_obsessive_flag", "psychoticism": "scl90_psychoticism_flag",
        "paranoid": "scl90_paranoid_flag", "hostility": "scl90_hostility_flag",
    }

    def classify_flags(self, flags: dict[str, int]) -> dict[str, object]:
        normalized = {key: int(flags.get(key, 0) == 1) for key in self.FEATURE_ORDER}
        vector = [normalized[key] for key in self.FEATURE_ORDER]
        count = sum(vector)
        high = (
            normalized["suicidal_ideation"] == 1
            or (normalized["upi_flag"] == 1 and count >= 3)
            or count >= 4
            or (normalized["upi_flag"] == 1 and (
                normalized["scl90_depression_flag"] == 1 or normalized["scl90_anxiety_flag"] == 1
            ))
        )
        return {
            "cluster_id": int(high),
            "cluster_name": "高心理负荷型" if high else "低心理负荷相对平稳型",
            "algorithm": "rule-k2-matching-study-features",
            "feature_order": list(self.FEATURE_ORDER), "feature_vector": vector,
            "positive_flag_count": count, "flags": normalized,
            "reason": "本地可解释规则二分类，仅为支持关注线索；不是训练所得的 KMeans，也不是临床诊断。",
        }

    def classify_assessment(self, assessment: dict[str, object]) -> dict[str, object]:
        # Original features remain local; never trust a stored prose label as model input.
        factors = assessment.get("factor_scores") or {}
        upi = factors.get("upi", {})
        if upi and all(key in factors for key in self.FACTOR_FLAGS):
            flags = {"upi_flag": int(bool(upi.get("positive"))),
                     "suicidal_ideation": int(bool(assessment.get("safety_signal")))}
            flags.update({flag: int(bool(factors[key].get("positive")))
                          for key, flag in self.FACTOR_FLAGS.items()})
            return self.classify_flags(flags)
        risk = assessment.get("risk_level", "unknown")
        return {
            "cluster_id": None,
            "cluster_name": {"high": "优先支持关注", "medium": "建议进一步关注",
                             "low": "日常支持关注"}.get(risk, "资料不足"),
            "algorithm": "custom-assessment-priority-v1",
            "reason": "自编或历史短测评不映射为标准量表或研究分群。",
        }

    def model_context(self, assessment: dict[str, object], safety_signal: bool = False) -> dict[str, object]:
        result = self.classify_assessment(assessment)
        safety = safety_signal or bool(assessment.get("safety_signal"))
        return {
            "cluster_id": result["cluster_id"], "cluster_name": result["cluster_name"],
            "algorithm": result["algorithm"],
            "attention_level": "high" if safety else assessment.get("risk_level", "unknown"),
            "safety_signal": safety,
        }
