from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any


class AgentKnowledgeService:
    """Reads the supplied Agent corpus as data, without executing its demo scripts."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.knowledge = directory / "knowledge"
        self.records = self._read_records()
        self.risk_rules = self._read_json("risk_rules.json", [])
        self.style_rules = self._read_json("report_style_rules.json", {})
        self.feature_order = self._read_feature_order()

    def _read_json(self, filename: str, fallback: Any) -> Any:
        try:
            return json.loads((self.knowledge / filename).read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            return fallback

    def _read_records(self) -> list[dict[str, str]]:
        path = self.knowledge / "psych_rag_kb.csv"
        try:
            with path.open(encoding="utf-8-sig", newline="") as handle:
                return [
                    {str(key): str(value or "") for key, value in row.items()}
                    for row in csv.DictReader(handle)
                    if str(row.get("content", "")).strip()
                ]
        except OSError:
            return []

    def _read_feature_order(self) -> list[str]:
        try:
            payload = json.loads(
                (self.directory / "models" / "feature_order.json").read_text(encoding="utf-8-sig")
            )
            return payload if isinstance(payload, list) else payload.get("feature_order", [])
        except (OSError, ValueError, AttributeError):
            return []

    def rules_for_report(self, safety_signal: bool) -> str:
        applicable = []
        for rule in self.risk_rules if isinstance(self.risk_rules, list) else []:
            trigger = rule.get("trigger", {})
            if trigger.get("always") or (
                safety_signal and trigger.get("suicidal_ideation") == 1
            ):
                selected = {
                    key: rule[key]
                    for key in ("rule_name", "must_include", "must_not_include", "style_constraints")
                    if key in rule
                }
                for key in ("must_include", "style_constraints"):
                    if isinstance(selected.get(key), list):
                        selected[key] = [line for line in selected[key] if not any(word in str(line) for word in ("聚类", "分群", "cluster"))]
                applicable.append(selected)
        tone = self.style_rules.get("tone", {}) if isinstance(self.style_rules, dict) else {}
        constraints = self.style_rules.get("writing_constraints", []) if isinstance(self.style_rules, dict) else []
        constraints = [line for line in constraints if not any(word in str(line) for word in ("聚类", "分群", "cluster"))]
        return json.dumps(
            {"tone": tone, "writing_constraints": constraints, "safety_rules": applicable},
            ensure_ascii=False,
        )

    def chat_guidance(self) -> str:
        return (
            "本地 Agent 知识库仅提供心理教育与支持信息，不是临床诊断依据。"
            "使用客观、支持性、非污名化语气；不得编造测评、聚类、学业或个人事实。"
            "若用户完成了 SCL-90 十因子与 UPI 筛查，可使用其阳性标记作研究性分群参考，仍不得写成诊断。"
        )

    def review_report(self, content: str, safety_signal: bool) -> str:
        replacements = {
            "你患有抑郁症": "现有资料提示情绪困扰，需要专业评估",
            "你患有焦虑症": "现有资料提示焦虑相关体验，需要专业评估",
            "你属于精神障碍患者": "不能根据现有资料作出临床判断",
            "你已经确诊": "现有资料不能用于确诊",
            "必须药物治疗": "是否需要治疗应由有资质的专业人员评估",
            "暂无即时危险迹象": "当前资料未触发预设安全规则，但不能据此排除风险",
            "风险可控": "尚无足够资料判断即时风险",
            "未达到临床诊断标准": "不能据此判断是否符合任何临床诊断标准",
            "整体心理负荷呈现轻度上升趋势": "当前资料提示压力相关体验，尚不足以判断变化趋势",
            "拨打 12356 寻求学校心理支持": "拨打 12356 寻求心理支持",
            "未发现即时危险信号或明确的自杀意念表达": "本次输入未触发预设安全规则，但这不能排除即时危险",
        }
        result = content.strip()
        for unsafe, safe in replacements.items():
            result = result.replace(unsafe, safe)
        result = re.sub(r"[^。\n]*(?:聚类结果反映|该分群|你属于.{0,12}型)[^。\n]*[。]?", "", result)
        boundary = "本报告仅用于辅助支持，不构成临床诊断结论。自动生成内容不能替代专业心理咨询或医学评估。"
        if boundary not in result:
            result += "\n\n## 提醒与边界\n" + boundary
        if safety_signal:
            result = (
                "## 优先安全支持\n"
                "当前存在较高优先级关注信号。请尽快联系可信任的人、学校心理中心或专业心理支持。"
                "可拨打 12356；若存在即时危险，请立即拨打 110/120 或前往急诊。\n\n"
                + result
            )
        return result

    def status(self) -> dict[str, object]:
        return {
            "available": bool(self.records),
            "records": len(self.records),
            "searchable_records": sum(1 for item in self.records if not item.get("applicable_clusters")),
            "source": "Agent/knowledge/psych_rag_kb.csv",
            "rules_loaded": bool(self.risk_rules),
            "cluster_model_enabled": True,
            "classification_algorithm": "rule-k2-matching-study-features",
            "trained_cluster_model_loaded": False,
            "cluster_model_reason": (
                "自编筛查的阳性标记先做本地规则二分类；大模型仅接收精简分类结果。未加载 joblib 研究模型。"
            ),
            "example_student_data_public": False,
        }
