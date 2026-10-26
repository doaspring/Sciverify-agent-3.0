"""DesignContext：研究设计事实的用户确认记录（V3 方案 §3.2/§4.3）.

铁律：
- 实验单位的独立性只能由用户根据研究设计确认，禁止从列名/数据形态推断；
- independence_confirmed=True 必须伴随 unit_semantics（用户对"单位代表什么"的回答）
  和 confirmed_by_user=True，否则视为未确认；
- Pydantic extra=forbid：任何未声明字段直接拒绝。
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

import pandas as pd
from pydantic import BaseModel, ConfigDict, field_validator, model_validator


class DesignType(str, Enum):
    INDEPENDENT_GROUPS = "independent_groups"  # 两组独立单位
    PAIRED = "paired"                          # 配对设计（需 pair_id 列）
    UNKNOWN = "unknown"                        # 尚未确认


class AggregationPolicy(str, Enum):
    NONE = "none"                # 无技术重复，行级即单位级
    MEAN_BY_UNIT = "mean_by_unit"  # 技术重复按单位取均值（用户授权后 M3 生效）


class DesignContext(BaseModel):
    """一次分析的设计事实；由 UI 交互构造，序列化进证据链。"""

    model_config = ConfigDict(extra="forbid")

    dataset_id: str
    group_column: str
    outcome_column: str
    independent_unit_column: str
    technical_repeat_column: str | None = None
    pair_id_column: str | None = None
    design_type: DesignType = DesignType.UNKNOWN
    aggregation_policy: AggregationPolicy = AggregationPolicy.NONE
    independence_confirmed: bool = False
    confirmed_by_user: bool = False
    unit_semantics: str = ""  # 用户回答：处理独立施加到什么对象上
    confirmation_timestamp: datetime | None = None

    @model_validator(mode="after")
    def _check_confirmation_integrity(self) -> "DesignContext":
        if self.independence_confirmed and not self.unit_semantics.strip():
            raise ValueError(
                "independence_confirmed=True 必须填写 unit_semantics"
                "（用户需回答实验单位代表什么，不允许盲目勾选确认）。"
            )
        if self.independence_confirmed and not self.confirmed_by_user:
            raise ValueError(
                "independence_confirmed=True 需要同时 confirmed_by_user=True"
                "（确认动作必须来自用户交互，不能由系统代答）。"
            )
        # 注意：配对设计缺 pair_id 不在 schema 层拒绝——由规则引擎
        # PAIR_ID_MISSING 产生可解释 BLOCK（UI 流程需要先构造再审查）。
        return self

    @field_validator("confirmation_timestamp")
    @classmethod
    def _ts_utc(cls, v: datetime | None) -> datetime | None:
        if v is not None and v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v

    def validate_against(self, df: pd.DataFrame) -> None:
        """校验所有已声明列确实存在于数据中；类型不匹配给明确报错。"""
        required = [self.group_column, self.outcome_column, self.independent_unit_column]
        optional = [self.technical_repeat_column, self.pair_id_column]
        for col in required + [c for c in optional if c]:
            if col not in df.columns:
                raise ValueError(
                    f"设计声明列 {col!r} 不在数据列 {list(df.columns)} 中。"
                )
        if not pd.api.types.is_numeric_dtype(df[self.outcome_column]):
            raise TypeError(
                f"结局列 {self.outcome_column!r} 必须数值型，"
                f"当前 {df[self.outcome_column].dtype}。"
            )

    def snapshot(self) -> dict:
        """进入证据链的确定性快照。"""
        return self.model_dump(mode="json")


def now_utc() -> datetime:
    return datetime.now(timezone.utc)
