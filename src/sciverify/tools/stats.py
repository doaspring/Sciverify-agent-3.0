"""白名单统计工具（确定性计算）：聚合、描述、Welch t、配对 t、箱线图.

铁律（V3 方案 §3.4/§4.5）：
- 一切数值由 Pandas/SciPy/NumPy 确定性计算；本模块不含任何 LLM 输出路径；
- 推断以"独立实验单位"为样本量：调用方必须传入单位级数据（或经授权聚合）；
- 有效样本量 = n_units，不是 n_rows；两者都进 ToolResult；
- 聚合只在用户显式授权（MEAN_BY_UNIT）后执行；不适用简单聚合的
  重复测量/层级设计必须给出明确"暂不支持"。
"""
from __future__ import annotations

import math
from typing import Any

import matplotlib

matplotlib.use("Agg")  # 无头渲染，Streamlit/服务器安全
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from pydantic import BaseModel, ConfigDict
from scipy import stats as sps

from sciverify.design.schema import AggregationPolicy, DesignContext, DesignType


class UnsupportedDesignError(ValueError):
    """当前版本明确不支持的设计/聚合形态（不擅自扁平化）。"""


def _library_versions() -> dict[str, str]:
    return {
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pandas": pd.__version__,
    }


# ================================================== 描述统计（基础）


def describe_groups(
    df: pd.DataFrame, group_col: str, value_col: str
) -> dict[str, dict[str, float | int]]:
    """按组描述统计：n / mean / std(ddof=1) / min / max（对传入行级数据）.

    注意：这里 n 是传入数据的行数。推断前请传入单位级数据，
    使 n 等于独立实验单位数（由 prepare_analysis_frame 保证）。
    """
    for col in (group_col, value_col):
        if col not in df.columns:
            raise KeyError(f"列 {col!r} 不存在")
    series = df[value_col]
    if not pd.api.types.is_numeric_dtype(series):
        raise TypeError(f"结局列 {value_col!r} 必须是数值类型，当前 {series.dtype}")

    out: dict[str, dict[str, float | int]] = {}
    for group_name, sub in df.groupby(group_col)[value_col]:
        vals = sub.dropna()
        n = int(vals.size)
        out[str(group_name)] = {
            "n": n,
            "mean": float(vals.mean()) if n else math.nan,
            "std": float(vals.std(ddof=1)) if n >= 2 else math.nan,
            "min": float(vals.min()) if n else math.nan,
            "max": float(vals.max()) if n else math.nan,
        }
    if len(out) < 2:
        raise ValueError(f"分组列 {group_col!r} 至少需要 2 个组，当前 {len(out)} 个")
    return out


# ================================================== 单位级聚合


def aggregate_by_unit(df: pd.DataFrame, design: DesignContext) -> pd.DataFrame:
    """按 unit × group 聚合技术重复为每单位一行（均值）。

    前置：用户已确认技术重复并授权 MEAN_BY_UNIT；否则抛错。
    输出保留原始列名（unit_col/group_col/value_col）+ n_tech_reps，
    使后续工具调用可直接使用设计声明的列名。
    """
    unit_col, group_col, value_col = (
        design.independent_unit_column,
        design.group_column,
        design.outcome_column,
    )
    rows_per_unit = df.groupby(unit_col, dropna=False).size()
    has_multi = bool((rows_per_unit > 1).any())

    if has_multi:
        if not design.technical_repeat_column:
            raise UnsupportedDesignError(
                "暂不支持：多个测量行共享同一实验单位，但未声明技术重复列。"
                "可能是重复测量/层级设计，本版本不自动扁平化；请先在设计中"
                "确认多行性质并授权聚合，或修正单位列。"
            )
        if design.aggregation_policy != AggregationPolicy.MEAN_BY_UNIT:
            raise UnsupportedDesignError(
                "暂不支持：存在每单位多行但聚合策略未授权为 MEAN_BY_UNIT；"
                "禁止未经确认直接按行分析。"
            )
        if not design.independence_confirmed:
            raise UnsupportedDesignError(
                "暂不支持：独立实验单位尚未确认，不能执行任何聚合或推断。"
            )

    grouped = (
        df.groupby([unit_col, group_col], dropna=False)[value_col]
        .agg(**{value_col: "mean", "n_tech_reps": "size"})
        .reset_index()
    )
    grouped["n_tech_reps"] = grouped["n_tech_reps"].astype(int)
    return grouped


def prepare_analysis_frame(df: pd.DataFrame, design: DesignContext) -> pd.DataFrame:
    """按设计准备分析用单位级数据；无多行时保持一行一单位。"""
    return aggregate_by_unit(df, design)


# ================================================== ToolResult 结构


class ToolResult(BaseModel):
    """统计工具的结构化输出（报告与核验的唯一数值来源）。"""

    model_config = ConfigDict(extra="forbid")

    tool: str
    n_rows_raw: int
    n_units_used: int
    n_by_group: dict[str, int]
    mean_by_group: dict[str, float]
    statistic: float
    p_value: float
    mean_difference: float
    confidence_interval: tuple[float, float] | None = None
    difference_definition: str = ""
    warnings: list[str] = []
    library_versions: dict[str, str] = {}


def _require_two_groups(unit_df: pd.DataFrame, group_col: str) -> list[str]:
    groups = sorted(unit_df[group_col].astype(str).unique())
    if len(groups) != 2:
        raise ValueError(f"组间比较需要恰好 2 个组，当前 {len(groups)} 个：{groups}")
    return groups


def _drop_missing(unit_df: pd.DataFrame, value_col: str) -> tuple[pd.DataFrame, list[str]]:
    warnings: list[str] = []
    n_missing = int(unit_df[value_col].isna().sum())
    if n_missing:
        warnings.append(
            f"单位级数据剔除 {n_missing} 个缺失结局值（策略：成对/成组剔除并记录）"
        )
        unit_df = unit_df.dropna(subset=[value_col])
    return unit_df, warnings


# ================================================== Welch 两组


def welch_t_test(
    unit_df: pd.DataFrame, group_col: str, value_col: str, alpha: float = 0.05
) -> ToolResult:
    """独立两组 Welch t 检验（unit_df 必须是单位级数据）。"""
    if alpha <= 0 or alpha >= 1:
        raise ValueError("alpha 必须在 (0,1) 区间")
    groups = _require_two_groups(unit_df, group_col)
    unit_df, warnings = _drop_missing(unit_df, value_col)

    samples = {}
    for g in groups:
        vals = unit_df.loc[unit_df[group_col].astype(str) == g, value_col].to_numpy(dtype=float)
        if vals.size < 2:
            raise ValueError(f"组 {g!r} 的独立单位数 {vals.size} < 2，无法执行 Welch 检验")
        samples[g] = vals

    g1, g2 = groups  # 排序后的前/后组
    a, b = samples[g1], samples[g2]
    # ttest_ind(b, a)：统计量符号与 mean_difference = m2 - m1 保持一致
    res = sps.ttest_ind(b, a, equal_var=False)

    m1, m2 = float(a.mean()), float(b.mean())
    v1, v2 = float(a.var(ddof=1)), float(b.var(ddof=1))
    n1, n2 = int(a.size), int(b.size)
    se = math.sqrt(v1 / n1 + v2 / n2)
    if se == 0:
        raise ValueError("两组方差均为 0（数值完全相同），检验无意义")
    df_welch = se**4 / (
        v1**2 / (n1**2 * (n1 - 1)) + v2**2 / (n2**2 * (n2 - 1))
    )
    t_crit = float(sps.t.ppf(1 - alpha / 2, df_welch))
    diff = m2 - m1  # 定义：后组 − 前组（按组名排序）

    if min(n1, n2) < 5:
        warnings.append(f"样本量偏小（{g1}: n={n1}, {g2}: n={n2}），检验功效有限")

    return ToolResult(
        tool="welch_t_test",
        n_rows_raw=int(len(unit_df)),
        n_units_used=n1 + n2,
        n_by_group={g1: n1, g2: n2},
        mean_by_group={g1: m1, g2: m2},
        statistic=float(res.statistic),
        p_value=float(res.pvalue),
        mean_difference=diff,
        confidence_interval=(diff - t_crit * se, diff + t_crit * se),
        difference_definition=f"mean[{g2}] - mean[{g1}]（组名按字典序）",
        warnings=warnings,
        library_versions=_library_versions(),
    )


# ================================================== 配对 t


def paired_t_test(
    unit_df: pd.DataFrame,
    group_col: str,
    value_col: str,
    pair_col: str,
    alpha: float = 0.05,
) -> ToolResult:
    """配对 t 检验：按 pair_id 严格匹配，禁止按行序配对。"""
    if alpha <= 0 or alpha >= 1:
        raise ValueError("alpha 必须在 (0,1) 区间")
    groups = _require_two_groups(unit_df, group_col)
    unit_df, warnings = _drop_missing(unit_df, value_col)

    wide = unit_df.pivot_table(
        index=pair_col, columns=group_col, values=value_col, aggfunc="first"
    )
    missing_pairs = wide[groups].isna().any(axis=1).sum()
    if missing_pairs:
        raise ValueError(
            f"{int(missing_pairs)} 个配对缺少某一条件的观测，配对结构不完整；"
            "禁止按行序补齐。"
        )
    wide = wide.sort_index()
    g1, g2 = groups
    d = (wide[g2] - wide[g1]).to_numpy(dtype=float)  # d = 后组 − 前组
    n = int(d.size)
    if n < 2:
        raise ValueError(f"有效配对数 {n} < 2，无法执行配对检验")
    if float(d.var(ddof=1)) == 0 and n >= 2:
        raise ValueError("配对差值方差为 0，检验无意义")

    res = sps.ttest_rel(wide[g2].to_numpy(dtype=float), wide[g1].to_numpy(dtype=float))
    mean_d = float(d.mean())
    se_d = float(d.std(ddof=1) / math.sqrt(n))
    t_crit = float(sps.t.ppf(1 - alpha / 2, n - 1))

    if n < 5:
        warnings.append(f"配对数偏小（n={n}），检验功效有限")

    return ToolResult(
        tool="paired_t_test",
        n_rows_raw=int(len(unit_df)),
        n_units_used=n,  # 有效样本量 = 配对数
        n_by_group={g1: n, g2: n},
        mean_by_group={g1: float(wide[g1].mean()), g2: float(wide[g2].mean())},
        statistic=float(res.statistic),
        p_value=float(res.pvalue),
        mean_difference=mean_d,
        confidence_interval=(mean_d - t_crit * se_d, mean_d + t_crit * se_d),
        difference_definition=f"mean[{g2}] - mean[{g1}]（组名按字典序，按 pair_id 匹配）",
        warnings=warnings,
        library_versions=_library_versions(),
    )


# ================================================== 描述统计（ToolResult 化）


def describe_groups_result(
    unit_df: pd.DataFrame, group_col: str, value_col: str
) -> dict[str, Any]:
    """按组描述统计（单位级）；保留原 dict 形态便于 JSON 序列化。"""
    out = describe_groups(unit_df, group_col, value_col)
    out["_library_versions"] = _library_versions()
    return out


# ================================================== 可视化


def draw_boxplot(
    unit_df: pd.DataFrame,
    group_col: str,
    value_col: str,
    out_path: str,
    title: str = "按实验单位的分组箱线图",
    subtitle: str = "",
) -> str:
    """单位级箱线图（matplotlib Agg，返回文件路径）。"""
    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=150)
    groups = sorted(unit_df[group_col].astype(str).unique())
    data = [
        unit_df.loc[unit_df[group_col].astype(str) == g, value_col].dropna().to_numpy()
        for g in groups
    ]
    ax.boxplot(data, showmeans=True)
    ax.set_xticklabels(groups)
    ax.set_ylabel(value_col)
    ax.set_title(title, fontsize=11)
    if subtitle:
        ax.set_xlabel(subtitle, fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
