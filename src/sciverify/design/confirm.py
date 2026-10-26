"""推断门控：未确认独立实验单位前，禁止任何组间推断性检验（M1 验收硬标准）.

M2 的风险规则引擎在其上游做设计审查；本模块是最小门控，保证即使规则
尚未覆盖，推断工具也无法在"一行=一独立单位"未获确认时执行。
"""
from __future__ import annotations

from sciverify.design.schema import DesignContext, DesignType

# 必须向用户提出的问题（V3 方案 §3.2 原文）
QUESTION_UNIT_MEANING = (
    "处理（菌株/工艺/处理条件）是独立施加到哪个对象上的"
    "（样本、发酵罐、批次、培养皿、动物等）？该对象的每一例是否构成一个"
    "独立实验单位？多次测量同一单位是否只是技术重复？"
)


class InferenceBlockedError(RuntimeError):
    """推断性检验被阻断：独立实验单位未获用户确认。"""


def assert_inference_allowed(design: DesignContext | None) -> None:
    """推断工具执行的最后一道门。

    Raises:
        InferenceBlockedError: design 缺失、类型未知或未确认独立性。
    """
    if design is None:
        raise InferenceBlockedError(
            "尚未建立设计确认（DesignContext），禁止执行组间推断性检验。"
            f"请先回答：{QUESTION_UNIT_MEANING}"
        )
    if design.design_type == DesignType.UNKNOWN:
        raise InferenceBlockedError(
            "研究设计类型未知（independent_groups / paired），"
            "禁止执行推断性检验。请先确认设计类型。"
        )
    if not (design.independence_confirmed and design.confirmed_by_user):
        raise InferenceBlockedError(
            "独立实验单位未经用户确认，禁止把测量行当作独立样本执行推断检验。"
            f"请先回答：{QUESTION_UNIT_MEANING}"
        )
