from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any
from .screening_classifier import ScreeningClassifier


@dataclass(frozen=True)
class AssessmentQuestion:
    id: str
    text: str
    factor: str
    scale: str = "scl"
    section: str = ""
    scored: bool = True


class AssessmentService:
    """SCL-90 十因子 + UPI 筛查计分（自我觉察，不是临床诊断）。"""

    SCL_CUTOFF = 2.5
    UPI_CUTOFF = 25
    SCL_OPTIONS = (
        {"value": 1, "label": "没有"},
        {"value": 2, "label": "很轻"},
        {"value": 3, "label": "中等"},
        {"value": 4, "label": "偏重"},
        {"value": 5, "label": "严重"},
    )
    UPI_OPTIONS = (
        {"value": 0, "label": "否"},
        {"value": 1, "label": "是"},
    )
    LEGACY_OPTIONS = (
        {"value": 0, "label": "从未"},
        {"value": 1, "label": "偶尔"},
        {"value": 2, "label": "经常"},
        {"value": 3, "label": "几乎每天"},
    )
    FACTOR_NAMES = {
        "somatization": "躯体化",
        "obsessive": "强迫症状",
        "interpersonal": "人际关系敏感",
        "depression": "抑郁",
        "anxiety": "焦虑",
        "hostility": "敌对",
        "phobic": "恐怖",
        "paranoid": "偏执",
        "psychoticism": "精神病性",
        "additional": "其他（睡眠与饮食）",
        "upi": "UPI 总体负荷",
        "safety": "安全信号",
    }
    FACTOR_TO_FLAG = {
        "depression": "scl90_depression_flag",
        "anxiety": "scl90_anxiety_flag",
        "interpersonal": "scl90_interpersonal_sensitivity_flag",
        "obsessive": "scl90_obsessive_flag",
        "psychoticism": "scl90_psychoticism_flag",
        "paranoid": "scl90_paranoid_flag",
        "hostility": "scl90_hostility_flag",
    }
    CLUSTER_FEATURES = (
        "upi_flag",
        "scl90_depression_flag",
        "scl90_anxiety_flag",
        "scl90_interpersonal_sensitivity_flag",
        "scl90_obsessive_flag",
        "scl90_psychoticism_flag",
        "scl90_paranoid_flag",
        "scl90_hostility_flag",
        "suicidal_ideation",
    )
    # 条目为自编筛查题，计分规则对齐公开文献中的 SCL-90 十因子均分与 UPI 总分。
    QUESTIONS = (
        AssessmentQuestion("scl_som_1", "最近一周，头痛、头昏或身体某些部位发麻、刺痛。", "somatization", "scl", "scl90"),
        AssessmentQuestion("scl_som_2", "最近一周，胸口发闷、心跳加快或呼吸不畅。", "somatization", "scl", "scl90"),
        AssessmentQuestion("scl_som_3", "最近一周，胃肠不适、恶心或身体发紧、酸痛。", "somatization", "scl", "scl90"),
        AssessmentQuestion("scl_ocd_1", "最近一周，脑子里反复出现同样想法，很难甩掉。", "obsessive", "scl", "scl90"),
        AssessmentQuestion("scl_ocd_2", "最近一周，做事必须反复检查或确认才放心。", "obsessive", "scl", "scl90"),
        AssessmentQuestion("scl_ocd_3", "最近一周，做事必须按固定次序，否则会明显不安。", "obsessive", "scl", "scl90"),
        AssessmentQuestion("scl_int_1", "最近一周，与人相处时容易感到不自在或被注意。", "interpersonal", "scl", "scl90"),
        AssessmentQuestion("scl_int_2", "最近一周，觉得自己不如别人，或担心别人对你评价不好。", "interpersonal", "scl", "scl90"),
        AssessmentQuestion("scl_int_3", "最近一周，感情容易被伤害，或觉得别人不理解你。", "interpersonal", "scl", "scl90"),
        AssessmentQuestion("scl_dep_1", "最近一周，提不起兴趣，感到心情低落或空虚。", "depression", "scl", "scl90"),
        AssessmentQuestion("scl_dep_2", "最近一周，觉得自己没有价值，或对未来很悲观。", "depression", "scl", "scl90"),
        AssessmentQuestion("scl_dep_3", "最近一周，精力下降、做什么都费力，或责备自己。", "depression", "scl", "scl90"),
        AssessmentQuestion("scl_anx_1", "最近一周，感到紧张、担心或心里发慌，难以放松。", "anxiety", "scl", "scl90"),
        AssessmentQuestion("scl_anx_2", "最近一周，容易坐立不安，或突然害怕却说不清原因。", "anxiety", "scl", "scl90"),
        AssessmentQuestion("scl_anx_3", "最近一周，担心不好的事情会发生，并难以停止这种担心。", "anxiety", "scl", "scl90"),
        AssessmentQuestion("scl_hos_1", "最近一周，容易发脾气，看什么都不顺眼。", "hostility", "scl", "scl90"),
        AssessmentQuestion("scl_hos_2", "最近一周，有想摔东西或与人争执的冲动。", "hostility", "scl", "scl90"),
        AssessmentQuestion("scl_hos_3", "最近一周，忍不住对别人不客气，或心里充满怒气。", "hostility", "scl", "scl90"),
        AssessmentQuestion("scl_pho_1", "最近一周，害怕空旷场所、人群或独自外出。", "phobic", "scl", "scl90"),
        AssessmentQuestion("scl_pho_2", "最近一周，因为害怕而回避某些场所、交通或社交场合。", "phobic", "scl", "scl90"),
        AssessmentQuestion("scl_pho_3", "最近一周，想到特定场景就会心慌、出汗或想立刻离开。", "phobic", "scl", "scl90"),
        AssessmentQuestion("scl_par_1", "最近一周，觉得别人不怀好意，或在议论、针对你。", "paranoid", "scl", "scl90"),
        AssessmentQuestion("scl_par_2", "最近一周，很难信任别人，觉得自己被占便宜或被监视。", "paranoid", "scl", "scl90"),
        AssessmentQuestion("scl_par_3", "最近一周，把别人的中性言行理解成敌意或隐瞒。", "paranoid", "scl", "scl90"),
        AssessmentQuestion("scl_psy_1", "最近一周，觉得有人能知道你的想法，或自己的想法不受控制。", "psychoticism", "scl", "scl90"),
        AssessmentQuestion("scl_psy_2", "最近一周，即使独处也感到有人在影响你，或听到别人听不到的声音。", "psychoticism", "scl", "scl90"),
        AssessmentQuestion("scl_psy_3", "最近一周，觉得现实有些不真实，或自己与周围世界隔开了。", "psychoticism", "scl", "scl90"),
        AssessmentQuestion("scl_add_1", "最近一周，入睡困难、早醒或睡眠很浅。", "additional", "scl", "scl90"),
        AssessmentQuestion("scl_add_2", "最近一周，食欲明显下降或暴饮暴食。", "additional", "scl", "scl90"),
        AssessmentQuestion("scl_add_3", "最近一周，感到自己有罪、睡不好或吃不好相互影响。", "additional", "scl", "scl90"),
        AssessmentQuestion("upi_1", "食欲下降。", "upi", "upi", "upi"),
        AssessmentQuestion("upi_2", "恶心、胃部不适。", "upi", "upi", "upi"),
        AssessmentQuestion("upi_3", "容易拉肚子或便秘。", "upi", "upi", "upi"),
        AssessmentQuestion("upi_4", "关注自己的心跳或呼吸。", "upi", "upi", "upi"),
        AssessmentQuestion("upi_5", "身体健康，几乎没有不适。", "upi_lie", "upi", "upi", False),
        AssessmentQuestion("upi_6", "经常头痛。", "upi", "upi", "upi"),
        AssessmentQuestion("upi_7", "脖子或肩膀发紧发酸。", "upi", "upi", "upi"),
        AssessmentQuestion("upi_8", "胸闷或胸口不舒服。", "upi", "upi", "upi"),
        AssessmentQuestion("upi_9", "身体容易疲劳。", "upi", "upi", "upi"),
        AssessmentQuestion("upi_10", "对性或身体变化感到困扰。", "upi", "upi", "upi"),
        AssessmentQuestion("upi_11", "睡眠不好。", "upi", "upi", "upi"),
        AssessmentQuestion("upi_12", "觉得自己变丑了或体型难看。", "upi", "upi", "upi"),
        AssessmentQuestion("upi_13", "思想不集中。", "upi", "upi", "upi"),
        AssessmentQuestion("upi_14", "记忆力下降。", "upi", "upi", "upi"),
        AssessmentQuestion("upi_15", "缺乏耐力，稍做就累。", "upi", "upi", "upi"),
        AssessmentQuestion("upi_16", "缺乏自信心。", "upi", "upi", "upi"),
        AssessmentQuestion("upi_17", "过于担心将来。", "upi", "upi", "upi"),
        AssessmentQuestion("upi_18", "情绪容易起伏。", "upi", "upi", "upi"),
        AssessmentQuestion("upi_19", "父母期望让你感到沉重。", "upi", "upi", "upi"),
        AssessmentQuestion("upi_20", "总觉得自己朝气蓬勃。", "upi_lie", "upi", "upi", False),
        AssessmentQuestion("upi_21", "对任何事都没有兴趣。", "upi", "upi", "upi"),
        AssessmentQuestion("upi_22", "觉得自己没有价值。", "upi", "upi", "upi"),
        AssessmentQuestion("upi_23", "觉得活着没意思。", "upi", "upi", "upi"),
        AssessmentQuestion("upi_24", "常常责备自己。", "upi", "upi", "upi"),
        AssessmentQuestion("upi_25", "有过不想活下去或伤害自己的想法。", "safety", "upi", "upi"),
        AssessmentQuestion("upi_26", "对任何人都没有感情。", "upi", "upi", "upi"),
        AssessmentQuestion("upi_27", "感到自卑。", "upi", "upi", "upi"),
        AssessmentQuestion("upi_28", "觉得别人轻视你。", "upi", "upi", "upi"),
        AssessmentQuestion("upi_29", "觉得别人在背后说你坏话。", "upi", "upi", "upi"),
        AssessmentQuestion("upi_30", "在乎别人的视线。", "upi", "upi", "upi"),
        AssessmentQuestion("upi_31", "觉得别人不理解你。", "upi", "upi", "upi"),
        AssessmentQuestion("upi_32", "觉得别人不同情你。", "upi", "upi", "upi"),
        AssessmentQuestion("upi_33", "过多地考虑别人怎么看你。", "upi", "upi", "upi"),
        AssessmentQuestion("upi_34", "不愿意与人来往。", "upi", "upi", "upi"),
        AssessmentQuestion("upi_35", "至今生活过得很顺利。", "upi_lie", "upi", "upi", False),
        AssessmentQuestion("upi_36", "自己的过去和家庭让你烦恼。", "upi", "upi", "upi"),
        AssessmentQuestion("upi_37", "对容貌或体态过于在意。", "upi", "upi", "upi"),
        AssessmentQuestion("upi_38", "为恋爱或亲密关系苦恼。", "upi", "upi", "upi"),
        AssessmentQuestion("upi_39", "对遇到的人难以信任。", "upi", "upi", "upi"),
        AssessmentQuestion("upi_40", "觉得别人不可靠。", "upi", "upi", "upi"),
        AssessmentQuestion("upi_41", "容易被人误解。", "upi", "upi", "upi"),
        AssessmentQuestion("upi_42", "过于猜疑。", "upi", "upi", "upi"),
        AssessmentQuestion("upi_43", "不愿意听别人说话。", "upi", "upi", "upi"),
        AssessmentQuestion("upi_44", "自己的心被别人左右。", "upi", "upi", "upi"),
        AssessmentQuestion("upi_45", "爱操心。", "upi", "upi", "upi"),
        AssessmentQuestion("upi_46", "父母不理解你。", "upi", "upi", "upi"),
        AssessmentQuestion("upi_47", "觉得自己有罪。", "upi", "upi", "upi"),
        AssessmentQuestion("upi_48", "对任何事都没有决断力。", "upi", "upi", "upi"),
        AssessmentQuestion("upi_49", "缺乏热情和积极性。", "upi", "upi", "upi"),
        AssessmentQuestion("upi_50", "至今信心十足、从不气馁。", "upi_lie", "upi", "upi", False),
        AssessmentQuestion("upi_51", "自己的头脑不好使。", "upi", "upi", "upi"),
        AssessmentQuestion("upi_52", "思想被别人干扰或控制。", "upi", "upi", "upi"),
        AssessmentQuestion("upi_53", "总注意周围的人。", "upi", "upi", "upi"),
        AssessmentQuestion("upi_54", "莫名其妙地不安。", "upi", "upi", "upi"),
        AssessmentQuestion("upi_55", "一个人的时候感到不安。", "upi", "upi", "upi"),
        AssessmentQuestion("upi_56", "莫名其妙地害怕。", "upi", "upi", "upi"),
        AssessmentQuestion("upi_57", "站到高处会害怕。", "upi", "upi", "upi"),
        AssessmentQuestion("upi_58", "害怕遇见人。", "upi", "upi", "upi"),
        AssessmentQuestion("upi_59", "觉得别人在注视自己。", "upi", "upi", "upi"),
        AssessmentQuestion("upi_60", "觉得别人在说自己的坏话。", "upi", "upi", "upi"),
    )
    LEGACY_REQUIRED = {"anxiety", "obsessive", "somatization", "safety"}
    FACTOR_GUIDANCE = {
        "somatization": {
            "meaning": "身体不适主诉较多。公开文献中，SCL-90 躯体化反映头痛、胸闷、胃肠不适等，但首先要排除躯体疾病。",
            "measures": [
                "若症状突然加重、持续或影响日常，先到医院排查身体原因。",
                "规律作息、减少咖啡因，每天做 10 分钟腹式呼吸或渐进肌肉放松。",
                "把“身体信号”记下来（时间、情境、强度），便于和医生或咨询师沟通。",
            ],
        },
        "obsessive": {
            "meaning": "反复想法或重复核对较突出。高校筛查中，强迫因子升高常见于完美主义与不确定性焦虑。",
            "measures": [
                "对重复核对做“推迟练习”：先等 5–10 分钟再决定是否再查一次。",
                "把任务拆成可完成的一小步，允许“足够好”而不是绝对正确。",
                "若重复行为每天占用大量时间，建议预约学校心理中心做评估。",
            ],
        },
        "interpersonal": {
            "meaning": "人际敏感升高，常见体验是自卑、怕被评价、与人相处不自在。",
            "measures": [
                "选择 1 位相对安全的同学或朋友，做短时间、低压力的真实交流。",
                "把“别人一定在看我”改写成可检验的句子，例如“我不确定，需要证据”。",
                "参加结构清楚的小组活动，比大型社交场合更容易起步。",
            ],
        },
        "depression": {
            "meaning": "兴趣下降、自我评价低或动力不足较突出。这是筛查信号，不能当成抑郁症诊断。",
            "measures": [
                "每天安排一件 10–20 分钟、以前能带来一点成就感的小事（行为激活）。",
                "固定起床与出门时间，白天接受自然光，避免昼夜颠倒。",
                "若持续两周以上或伴轻生念头，立即联系可信任的人和专业支持。",
            ],
        },
        "anxiety": {
            "meaning": "紧张、担忧、坐立不安较突出。研究中焦虑与高认知负荷任务压力常同时出现。",
            "measures": [
                "练习 4-6 呼吸：吸气 4 秒、呼气 6 秒，连续 2 分钟。",
                "把担心写进“忧虑时间”（每天固定 15 分钟），其余时间先记下再放下。",
                "学习任务按 25 分钟一块推进，降低启动难度。",
            ],
        },
        "hostility": {
            "meaning": "易怒、争执冲动或看什么都不顺眼。敌对升高会伤害关系和自我调节。",
            "measures": [
                "感到怒气上升时先离开现场 10 分钟，用冷水洗脸或快走。",
                "用“我感到……因为……我需要……”代替指责句。",
                "保证睡眠；睡眠不足会明显放大怒气。",
            ],
        },
        "phobic": {
            "meaning": "对特定场所、人群或外出的害怕与回避较明显。",
            "measures": [
                "不要一次性强迫自己面对最怕的情境；从 1–10 分里选 3 分难度开始。",
                "每次短暂停留并等到身体反应回落，再结束，而不是一害怕就永久回避。",
                "若回避已经影响上课或出行，寻求咨询师做分级暴露指导。",
            ],
        },
        "paranoid": {
            "meaning": "猜疑、难信任或感到被针对。筛查阳性不等于偏执型障碍。",
            "measures": [
                "把怀疑写成“另一种可能”，至少列出两个非恶意解释。",
                "先向信任的人核实事实，避免只在想象里反复推演。",
                "减少熬夜和孤立，这两项都会加重猜疑感。",
            ],
        },
        "psychoticism": {
            "meaning": "现实感、思维被影响或感知异常相关条目偏高。这一项尤其需要专业评估，不能自行下结论。",
            "measures": [
                "尽快联系学校心理中心或精神科门诊，做面对面评估。",
                "保证睡眠，暂停可能加重混乱感的熬夜、独处反刍。",
                "请家人或朋友陪同，不要一个人硬扛。",
            ],
        },
        "additional": {
            "meaning": "睡眠或饮食问题较突出。附加因子常与压力和其他情绪因子一起升高。",
            "measures": [
                "固定起床时间；白天不补长觉；睡前 1 小时离开屏幕。",
                "一日三餐尽量规律，避免用完全不吃或暴食来调节情绪。",
                "若失眠超过 2 周或体重明显变化，同时看心身科或校医。",
            ],
        },
        "upi": {
            "meaning": "UPI 总分超过关注界值，表示近一年来多种身心不适条目较多，属于高校常用的一类关注线索。",
            "measures": [
                "建议预约学校心理中心做一次访谈，而不是只看分数。",
                "先稳住睡眠、饮食和一门最重要的课，避免全面停摆。",
                "把最困扰的 3 件事写下来，咨询时优先讨论。",
            ],
        },
    }
    REFERENCES = (
        "Derogatis, L. R. Symptom Checklist-90：条目 1–5 分，因子分=该因子条目均分。",
        "中文应用常把 10 个因子计为：躯体化、强迫、人际关系敏感、抑郁、焦虑、敌对、恐怖、偏执、精神病性、其他（睡眠与饮食）。",
        "国内筛查实践中，因子分≥2 或均分+1 个标准差都曾被使用；本系统按产品约定以因子分>2.5 作为关注提示。",
        "UPI（大学生人格问卷，樊富珉等引入高校）：56 个症状题计 1 分，测伪题不计分；一类关注常用总分≥25 或关键题阳性。本系统按约定以 UPI>25 作为关注提示。",
        "分群沿用本地研究特征：UPI 阳性、部分 SCL-90 因子阳性、自杀意念标记，区分为低心理负荷相对平稳型 / 高心理负荷型，仅作研究性归类。",
    )

    @classmethod
    def validate_questions(cls, questions: list[dict[str, Any]]) -> None:
        if not 4 <= len(questions) <= 40:
            raise ValueError("测评题目数量需为 4 到 40 道")
        ids = [str(item.get("id", "")).strip() for item in questions]
        if any(not item for item in ids) or len(ids) != len(set(ids)):
            raise ValueError("题目 id 不能为空且不能重复")
        factors = {str(item.get("factor", "")) for item in questions}
        if not cls.LEGACY_REQUIRED.issubset(factors):
            raise ValueError("测评必须包含焦虑、强迫、躯体化和安全信号题目")
        if any(str(item.get("factor", "")) not in cls.LEGACY_REQUIRED for item in questions):
            raise ValueError("题目 factor 仅支持 anxiety、obsessive、somatization、safety")
        if any(not str(item.get("text", "")).strip() for item in questions):
            raise ValueError("题目内容不能为空")

    def questions(self, template: dict[str, Any] | None = None) -> dict[str, object]:
        if template:
            questions = template["questions"]
            self.validate_questions(questions)
            return {
                "id": template["id"],
                "title": template["title"],
                "description": template["description"],
                "questions": questions,
                "options": list(self.LEGACY_OPTIONS),
                "instrument": "legacy",
                "disclaimer": "该测评仅用于自我觉察与人工关注线索，不构成医学诊断。",
            }
        return {
            "id": None,
            "title": "SCL-90 十因子与 UPI 筛查",
            "description": (
                "第一部分按 SCL-90 十因子计分（1–5 分，因子均分>2.5 为关注提示）；"
                "第二部分为 UPI 是/否题（症状题计分，UPI>25 为关注提示）。结果不是医学诊断。"
            ),
            "questions": [asdict(item) for item in self.QUESTIONS],
            "options": list(self.SCL_OPTIONS),
            "options_scl": list(self.SCL_OPTIONS),
            "options_upi": list(self.UPI_OPTIONS),
            "instrument": "scl90_upi",
            "cutoffs": {"scl90_factor": self.SCL_CUTOFF, "upi_total": self.UPI_CUTOFF},
            "disclaimer": "筛查分数只用于自我觉察和预约专业支持，不构成诊断或治疗决定。",
        }

    @staticmethod
    def _legacy_factor_label(percent: int) -> str:
        if percent >= 75:
            return "优先关注"
        if percent >= 50:
            return "需要关注"
        if percent >= 25:
            return "有些波动"
        return "状态平稳"

    def _score_legacy(self, answers: dict[str, int], questions: list[dict[str, Any]]) -> dict[str, object]:
        question_ids = {str(item["id"]) for item in questions}
        if set(answers) != question_ids:
            raise ValueError(f"请完成全部 {len(questions)} 道题")
        if any(not isinstance(value, int) or value < 0 or value > 3 for value in answers.values()):
            raise ValueError("每道题的分值必须为 0 到 3")
        factor_scores: dict[str, dict[str, object]] = {}
        for factor in ("anxiety", "obsessive", "somatization"):
            factor_ids = [str(item["id"]) for item in questions if item["factor"] == factor]
            raw = sum(answers[item_id] for item_id in factor_ids)
            maximum = len(factor_ids) * 3
            percent = round(raw / maximum * 100) if maximum else 0
            factor_scores[factor] = {
                "name": {"anxiety": "焦虑", "obsessive": "强迫", "somatization": "躯体化"}[factor],
                "raw": raw,
                "max": maximum,
                "mean": round(raw / len(factor_ids), 2) if factor_ids else 0,
                "percent": percent,
                "positive": percent >= 75,
                "label": self._legacy_factor_label(percent),
            }
        safety_ids = [str(item["id"]) for item in questions if item["factor"] == "safety"]
        safety_signal = any(answers[item_id] > 0 for item_id in safety_ids)
        highest = max(int(item["percent"]) for item in factor_scores.values())
        if safety_signal or highest >= 75:
            risk = "high"
            summary = "近期状态需要优先关注。请尽快联系可信任的人或专业支持；如有即时危险，请拨打 110/120 或心理援助热线 12356。"
        elif highest >= 50:
            risk = "medium"
            summary = "部分状态信号较明显，建议减少独自承受，安排休息并考虑联系专业支持。"
        else:
            risk = "low"
            summary = "目前未出现高优先级信号，仍建议持续记录变化并照顾好睡眠与日常节律。"
        if int(factor_scores["somatization"]["percent"]) >= 50:
            summary += " 若身体不适持续、突然加重或影响日常生活，请及时就医排查身体原因。"
        return {
            "score": sum(answers.values()),
            "max_score": len(questions) * 3,
            "risk_level": risk,
            "safety_signal": safety_signal,
            "summary": summary,
            "factor_scores": factor_scores,
            "elevated_factors": [],
            "classification": {
                "cluster_id": None,
                "cluster_name": "未启用",
                "reason": "当前为管理员自定义短量表，不能作为 UPI/SCL-90 分群输入。",
                "flags": {},
            },
            "disclaimer": "分数仅用于自我觉察和人工关注线索，不是临床量表，不构成诊断或治疗建议。",
        }

    def classify_screening(self, flags: dict[str, int]) -> dict[str, object]:
        return ScreeningClassifier().classify_flags(flags)

    def _score_scl_upi(self, answers: dict[str, int], questions: list[dict[str, Any]]) -> dict[str, object]:
        question_ids = {str(item["id"]) for item in questions}
        if set(answers) != question_ids:
            raise ValueError(f"请完成全部 {len(questions)} 道题")
        for item in questions:
            value = answers[str(item["id"])]
            scale = str(item.get("scale", "scl"))
            if scale == "upi":
                if not isinstance(value, int) or value not in (0, 1):
                    raise ValueError("UPI 题目只能选 否/是")
            elif not isinstance(value, int) or value < 1 or value > 5:
                raise ValueError("SCL-90 题目分值必须为 1 到 5")

        factor_scores: dict[str, dict[str, object]] = {}
        scl_factors = (
            "somatization",
            "obsessive",
            "interpersonal",
            "depression",
            "anxiety",
            "hostility",
            "phobic",
            "paranoid",
            "psychoticism",
            "additional",
        )
        for factor in scl_factors:
            ids = [str(item["id"]) for item in questions if item["factor"] == factor]
            mean = round(sum(answers[item_id] for item_id in ids) / len(ids), 2) if ids else 0.0
            positive = mean > self.SCL_CUTOFF
            factor_scores[factor] = {
                "name": self.FACTOR_NAMES[factor],
                "raw": sum(answers[item_id] for item_id in ids),
                "max": len(ids) * 5,
                "item_count": len(ids),
                "mean": mean,
                "percent": round((mean - 1) / 4 * 100) if ids else 0,
                "positive": positive,
                "cutoff": self.SCL_CUTOFF,
                "label": f"因子分 {mean}，{'高于' if positive else '未高于'} {self.SCL_CUTOFF} 关注线",
            }

        upi_ids = [
            str(item["id"])
            for item in questions
            if item.get("scored", True) and item.get("scale") == "upi" and item["factor"] != "upi_lie"
        ]
        upi_total = sum(int(answers[item_id]) for item_id in upi_ids)
        upi_positive = upi_total > self.UPI_CUTOFF
        factor_scores["upi"] = {
            "name": self.FACTOR_NAMES["upi"],
            "raw": upi_total,
            "max": len(upi_ids),
            "item_count": len(upi_ids),
            "mean": upi_total,
            "percent": round(upi_total / len(upi_ids) * 100) if upi_ids else 0,
            "positive": upi_positive,
            "cutoff": self.UPI_CUTOFF,
            "label": f"UPI {upi_total}/{len(upi_ids)}，{'高于' if upi_positive else '未高于'} {self.UPI_CUTOFF} 关注线",
        }

        safety_ids = [str(item["id"]) for item in questions if item["factor"] == "safety"]
        safety_signal = any(int(answers[item_id]) > 0 for item_id in safety_ids)
        flags = {
            "upi_flag": int(upi_positive),
            "suicidal_ideation": int(safety_signal),
        }
        for factor, flag_name in self.FACTOR_TO_FLAG.items():
            flags[flag_name] = int(bool(factor_scores[factor]["positive"]))
        classification = self.classify_screening(flags)

        elevated = []
        for key, payload in factor_scores.items():
            if not payload["positive"] or key not in self.FACTOR_GUIDANCE:
                continue
            guide = self.FACTOR_GUIDANCE[key]
            elevated.append(
                {
                    "key": key,
                    "name": payload["name"],
                    "score_text": payload["label"],
                    "meaning": guide["meaning"],
                    "measures": list(guide["measures"]),
                }
            )

        positive_names = [item["name"] for item in elevated]
        if safety_signal:
            risk = "high"
            summary = (
                "出现自我伤害或轻生相关条目。这不是诊断，但属于优先安全信号。"
                "请立即联系可信任的人、学校心理中心；可拨打 12356。若有即时危险请拨打 110/120。"
            )
        elif upi_positive or sum(1 for item in factor_scores.values() if item["positive"]) >= 3:
            risk = "high"
            summary = (
                f"UPI 或多项 SCL-90 因子超过关注线（因子分>2.5，UPI>25）。"
                f"{'偏高项目：' + '、'.join(positive_names) + '。' if positive_names else ''}"
                "建议尽快预约专业心理咨询，不要把筛查分数当成病名。"
            )
        elif positive_names:
            risk = "medium"
            summary = (
                f"以下因子超过关注线：{'、'.join(positive_names)}。"
                "下面给出的是心理教育建议，需要时再由专业人员评估。"
            )
        else:
            risk = "low"
            summary = "本次筛查未出现因子分>2.5 或 UPI>25 的关注标记。仍建议保持睡眠与求助渠道畅通。"

        if factor_scores["somatization"]["positive"]:
            summary += " 躯体化偏高时，请同步排查身体原因。"

        return {
            "score": upi_total,
            "max_score": len(upi_ids),
            "scl90_cutoff": self.SCL_CUTOFF,
            "upi_cutoff": self.UPI_CUTOFF,
            "risk_level": risk,
            "safety_signal": safety_signal,
            "summary": summary,
            "factor_scores": factor_scores,
            "elevated_factors": elevated,
            "classification": classification,
            "references": list(self.REFERENCES),
            "disclaimer": (
                "本工具用自编条目按 SCL-90 十因子均分与 UPI 总分规则做筛查提示，"
                "不能替代 SCL-90 原量表或精神科面诊，不构成诊断。"
            ),
        }

    def score(
        self,
        answers: dict[str, int],
        questions: list[dict[str, Any]] | None = None,
    ) -> dict[str, object]:
        active_questions = questions or [asdict(item) for item in self.QUESTIONS]
        scales = {str(item.get("scale", "")) for item in active_questions}
        if "scl" in scales or "upi" in scales:
            return self._score_scl_upi(answers, active_questions)
        self.validate_questions(active_questions)
        return self._score_legacy(answers, active_questions)
