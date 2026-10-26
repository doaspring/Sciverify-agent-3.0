"""科研统计防错引擎：六条设计风险规则（P0 核心创新，V3 方案 §3.3）.

铁律：
- 规则引擎只报告与阻断，绝不"自动修正"数据或设计；
- 同一 unit_id 的多行不自动视为技术重复，必须结合用户确认；
- 所有 finding 携带可核查证据（列名、单位计数、行计数）。

规则表：
  EU_UNKNOWN          未确认独立实验单位 → BLOCK
  TECH_REP_RISK       多行共享 unit 且未确认技术重复 → BLOCK（确认+授权聚合后解除）
  GROUP_UNIT_CONFLICT 独立两组声明下同单位跨组 → BLOCK
  PAIR_ID_MISSING     配对设计缺可靠 pair_id → BLOCK（禁止按行序配对）
  TOO_FEW_UNITS       任一组独立单位 <2 → BLOCK；<5 → WARN
  BAD_OUTCOME         结局非数值/组不完整/缺失严重 → BLOCK 或 WARN
"""
from __future__ import annotations

from enum import Enum

import pandas as pd
from pydantic import BaseModel, ConfigDict

from sciverify.design.schema import AggregationPolicy, DesignContext, DesignType

MIN_UNITS_BLOCK = 2   # 每组独立单位 <2 直接阻断该组间方法
MIN_UNITS_WARN = 5    # 2≤n<5 允许分析但警告
MISSING_BLOCK_RATE = 0.3


class Severity(str, Enum):
    PASS = "PASS"
    WARN = "WARN"
    BLOCK = "BLOCK"


class Resolution(str, Enum):
    OPEN = "OPEN"          # 待用户处理
    RESOLVED = "RESOLVED"  # 用户已确认/授权


class RiskFinding(BaseModel):
    """单条风险发现（进入证据链）。"""

    model_config = ConfigDict(extra="forbid")

    rule_id: str
    severity: Severity
    evidence_columns: list[str]
    affected_unit_count: int
    n_rows_involved: int = 0
    explanation: str
    required_action: str
    resolution_status: Resolution = Resolution.OPEN


class ReviewReport(BaseModel):
    """一次设计审查的完整结果。"""

    model_config = ConfigDict(extra="forbid")

    findings: list[RiskFinding]
    n_rows: int
    n_units: int
    n_units_by_group: dict[str, int]

    @property
    def overall(self) -> Severity:
        worst = Severity.PASS
        for f in self.findings:
            if f.resolution_status == Resolution.RESOLVED:
                continue
            if f.severity == Severity.BLOCK:
                return Severity.BLOCK
            if f.severity == Severity.WARN:
                worst = Severity.WARN
        return worst

    @property
    def open_blockers(self) -> list[RiskFinding]:
        return [
            f
            for f in self.findings
            if f.severity == Severity.BLOCK and f.resolution_status == Resolution.OPEN
        ]


# ---------------------------------------------------------------- 规则实现


def check_eu_unknown(design: DesignContext) -> RiskFinding | None:
    if design.independence_confirmed and design.confirmed_by_user:
        return None
    if design.design_type == DesignType.UNKNOWN:
        expl = "研究设计类型未确认（independent_groups/paired）。"
    else:
        expl = "独立实验单位未经用户确认，不能把测量行当作独立样本。"
    return RiskFinding(
        rule_id="EU_UNKNOWN",
        severity=Severity.BLOCK,
        evidence_columns=[design.independent_unit_column],
        affected_unit_count=0,
        explanation=expl,
        required_action="回答：处理独立施加到哪个对象上？该对象每例是否为独立实验单位？",
    )


def check_tech_rep_risk(df: pd.DataFrame, design: DesignContext) -> RiskFinding | None:
    if design.design_type == DesignType.PAIRED:
        # 配对设计中同一单位跨条件的重复测量就是配对结构本身，
        # 由 PAIR_ID_MISSING 规则负责校验，不按技术重复处理。
        return None
    unit_col = design.independent_unit_column
    rows_per_unit = df.groupby(unit_col, dropna=False).size()
    dup_units = rows_per_unit[rows_per_unit > 1]
    if dup_units.empty:
        return None  # 行级 == 单位级，无技术重复风险
    n_dup = int(dup_units.size)
    n_rows = int(dup_units.sum())

    confirmed = (
        design.technical_repeat_column
        and design.aggregation_policy == AggregationPolicy.MEAN_BY_UNIT
        and design.independence_confirmed
    )
    if confirmed:
        return RiskFinding(
            rule_id="TECH_REP_RISK",
            severity=Severity.PASS,
            evidence_columns=[unit_col, design.technical_repeat_column],
            affected_unit_count=n_dup,
            n_rows_involved=n_rows,
            explanation=(
                f"{n_dup} 个单位共 {n_rows} 行已由用户确认为技术重复，"
                "并授权按单位均值聚合（有效样本量 = 单位数）。"
            ),
            required_action="无需操作；分析将按 unit×group 聚合执行。",
            resolution_status=Resolution.RESOLVED,
        )
    return RiskFinding(
        rule_id="TECH_REP_RISK",
        severity=Severity.BLOCK,
        evidence_columns=[unit_col],
        affected_unit_count=n_dup,
        n_rows_involved=n_rows,
        explanation=(
            f"{n_dup} 个独立单位各有超过 1 行测量（涉及 {n_rows} 行）。"
            "系统不自动认定这些多行为技术重复——直接按行推断会高估样本量、"
            "低估方差（伪重复）。"
        ),
        required_action=(
            "请确认多行性质：若为同一单位的技术重复，请声明技术重复列并"
            "授权『按单位均值聚合』；若为重复测量/层级设计，当前版本不支持，"
            "请勿按行分析。"
        ),
    )


def check_group_unit_conflict(
    df: pd.DataFrame, design: DesignContext
) -> RiskFinding | None:
    if design.design_type != DesignType.INDEPENDENT_GROUPS:
        return None  # 配对设计允许同一单位跨条件出现
    unit_col, group_col = design.independent_unit_column, design.group_column
    groups_per_unit = df.groupby(unit_col, dropna=False)[group_col].nunique()
    conflict = groups_per_unit[groups_per_unit > 1]
    if conflict.empty:
        return None
    return RiskFinding(
        rule_id="GROUP_UNIT_CONFLICT",
        severity=Severity.BLOCK,
        evidence_columns=[unit_col, group_col],
        affected_unit_count=int(conflict.size),
        n_rows_involved=int(df[unit_col].isin(conflict.index).sum()),
        explanation=(
            f"{int(conflict.size)} 个单位在两组中都出现，与『独立两组』声明冲突："
            "可能是配对设计（同一单位先后两种条件）或单位列设置错误。"
        ),
        required_action="核实设计类型：若为配对设计请切换为 paired 并提供 pair_id；"
        "若单位列错误请重新确认 unit 列。",
    )


def check_pair_id_missing(df: pd.DataFrame, design: DesignContext) -> RiskFinding | None:
    if design.design_type != DesignType.PAIRED:
        return None
    pair_col = design.pair_id_column
    group_col = design.group_column
    if not pair_col:
        return RiskFinding(
            rule_id="PAIR_ID_MISSING",
            severity=Severity.BLOCK,
            evidence_columns=[],
            affected_unit_count=0,
            explanation="配对比较需要可靠的配对标识（pair_id），当前未提供。"
                        "系统禁止按行顺序拼接配对——行序配对几乎必然错配。",
            required_action="上传/指定含 pair_id 的列；若原始记录无配对标识，"
            "该数据不支持配对分析。",
        )
    if pair_col not in df.columns:
        return RiskFinding(
            rule_id="PAIR_ID_MISSING",
            severity=Severity.BLOCK,
            evidence_columns=[pair_col],
            affected_unit_count=0,
            explanation=f"配对列 {pair_col!r} 不在数据中。",
            required_action="重新选择真实存在的配对标识列。",
        )
    nulls = int(df[pair_col].isna().sum())
    if nulls:
        return RiskFinding(
            rule_id="PAIR_ID_MISSING",
            severity=Severity.BLOCK,
            evidence_columns=[pair_col],
            affected_unit_count=nulls,
            explanation=f"配对列存在 {nulls} 个空值，无法可靠配对。",
            required_action="补全配对标识或删除无法配对的记录（需记录删除规则）。",
        )
    # 每个配对在每组应恰好出现一次
    counts = df.groupby([pair_col, group_col]).size()
    bad = counts[counts != 1]
    if not bad.empty:
        return RiskFinding(
            rule_id="PAIR_ID_MISSING",
            severity=Severity.BLOCK,
            evidence_columns=[pair_col, group_col],
            affected_unit_count=int(bad.reset_index()[pair_col].nunique()),
            explanation=f"{len(bad)} 个 (pair, group) 组合的行数≠1，配对结构不完整。",
            required_action="核对配对标识：每个配对在每个条件下应恰有一条记录。",
        )
    return None


def check_too_few_units(df: pd.DataFrame, design: DesignContext) -> RiskFinding | None:
    unit_col, group_col = design.independent_unit_column, design.group_column
    n_by_group = df.groupby(group_col)[unit_col].nunique()
    if (n_by_group < MIN_UNITS_BLOCK).any():
        small = {str(g): int(n) for g, n in n_by_group[n_by_group < MIN_UNITS_BLOCK].items()}
        return RiskFinding(
            rule_id="TOO_FEW_UNITS",
            severity=Severity.BLOCK,
            evidence_columns=[unit_col, group_col],
            affected_unit_count=int((n_by_group < MIN_UNITS_BLOCK).sum()),
            explanation=(
                f"每组独立单位数不足 {MIN_UNITS_BLOCK}：{small}。"
                "即使测量很多行，单位数不足也无法做该版本的组间推断。"
            ),
            required_action="补充独立单位的实验数据，或仅做描述统计。",
        )
    if (n_by_group < MIN_UNITS_WARN).any():
        small = {str(g): int(n) for g, n in n_by_group[n_by_group < MIN_UNITS_WARN].items()}
        return RiskFinding(
            rule_id="TOO_FEW_UNITS",
            severity=Severity.WARN,
            evidence_columns=[unit_col, group_col],
            affected_unit_count=int((n_by_group < MIN_UNITS_WARN).sum()),
            explanation=f"以下组独立单位数偏少（<{MIN_UNITS_WARN}）：{small}；检验功效有限。",
            required_action="可在了解低功效风险下继续，或增加样本。",
        )
    return None


def check_bad_outcome(df: pd.DataFrame, design: DesignContext) -> RiskFinding | None:
    value_col, group_col = design.outcome_column, design.group_column
    if df[group_col].nunique() < 2:
        return RiskFinding(
            rule_id="BAD_OUTCOME",
            severity=Severity.BLOCK,
            evidence_columns=[group_col],
            affected_unit_count=0,
            explanation=f"分组列只有 {df[group_col].nunique()} 个组，无法组间比较。",
            required_action="确认分组列或补充对照组数据。",
        )
    if not pd.api.types.is_numeric_dtype(df[value_col]):
        return RiskFinding(
            rule_id="BAD_OUTCOME",
            severity=Severity.BLOCK,
            evidence_columns=[value_col],
            affected_unit_count=0,
            explanation=f"结局列 {value_col!r} 非数值型（{df[value_col].dtype}）。",
            required_action="选择数值结局列，或修正数据录入。",
        )
    missing = int(df[value_col].isna().sum())
    if missing:
        rate = missing / len(df)
        if rate > MISSING_BLOCK_RATE:
            return RiskFinding(
                rule_id="BAD_OUTCOME",
                severity=Severity.BLOCK,
                evidence_columns=[value_col],
                affected_unit_count=missing,
                explanation=f"结局缺失率 {rate:.0%}（{missing}/{len(df)}），过高。",
                required_action="补全数据或明确缺失机制后再分析。",
            )
        return RiskFinding(
            rule_id="BAD_OUTCOME",
            severity=Severity.WARN,
            evidence_columns=[value_col],
            affected_unit_count=missing,
            explanation=f"结局列有 {missing} 个缺失值，将按明确策略处理并记录。",
            required_action="确认缺失处理策略（默认成对剔除并写入报告）。",
        )
    return None


# ---------------------------------------------------------------- 入口


def review_design(df: pd.DataFrame, design: DesignContext) -> ReviewReport:
    """按固定顺序执行六条规则，返回审查报告（纯函数，无副作用）。"""
    checks = [
        check_eu_unknown(design),
        check_tech_rep_risk(df, design),
        check_group_unit_conflict(df, design),
        check_pair_id_missing(df, design),
        check_too_few_units(df, design),
        check_bad_outcome(df, design),
    ]
    findings = [f for f in checks if f is not None]
    n_units_by_group = {
        str(g): int(n)
        for g, n in df.groupby(design.group_column)[design.independent_unit_column].nunique().items()
    }
    return ReviewReport(
        findings=findings,
        n_rows=int(len(df)),
        n_units=int(df[design.independent_unit_column].nunique()),
        n_units_by_group=n_units_by_group,
    )
