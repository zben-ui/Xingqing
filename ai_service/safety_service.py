from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
import re
import unicodedata


SafetyLevel = Literal["normal", "emotional_support", "crisis"]


@dataclass(frozen=True)
class SafetyResult:
    level: SafetyLevel
    matched_keywords: tuple[str, ...] = ()


class SafetyService:
    CRISIS_KEYWORDS = (
        "不想活了", "不想活", "想死", "去死", "自杀", "轻生", "寻死",
        "结束生命", "活不下去", "没有活着的意义", "割腕", "跳楼", "吞药",
        "伤害自己", "杀了自己", "伤害别人", "杀人", "杀了他", "杀了她",
    )
    EMOTIONAL_KEYWORDS = (
        "焦虑", "难过", "压力大", "崩溃", "孤独", "害怕", "恐慌", "失眠",
        "委屈", "痛苦", "无助", "绝望", "撑不住", "很累", "好累", "烦躁",
    )

    def detect(self, text: str) -> SafetyResult:
        normalized = unicodedata.normalize("NFKC", text).lower().strip()
        # Detect common spacing/punctuation obfuscation, without diagnosing intent.
        normalized = re.sub(r"[\s\u200b\u200c\u200d\ufeff·，。！？、,.!?_-]+", "", normalized)
        crisis_matches = tuple(word for word in self.CRISIS_KEYWORDS if word in normalized)
        if crisis_matches:
            return SafetyResult("crisis", crisis_matches)
        emotional_matches = tuple(word for word in self.EMOTIONAL_KEYWORDS if word in normalized)
        if emotional_matches:
            return SafetyResult("emotional_support", emotional_matches)
        return SafetyResult("normal")

    @staticmethod
    def crisis_response() -> str:
        return (
            "听到你这样说，我很担心你现在的安全。请先不要独自承受，也先远离可能让你受伤的物品或地方，"
            "马上联系一位你信任的人——家人、朋友、老师或学校心理中心，请他们现在陪在你身边。"
            "你也可以拨打全国心理援助热线 12356。"
            "如果你觉得自己或他人可能马上受到伤害，请立即拨打 110/120 或前往最近的急诊。"
            "我可以继续陪你把求助的话组织出来，但此刻现实中的支持最重要。"
        )
