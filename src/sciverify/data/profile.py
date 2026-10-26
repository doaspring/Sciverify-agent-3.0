"""数据画像：行列规模、类型、缺失率、唯一值（M0 最小实现，M1 扩展设计字段映射）."""
from __future__ import annotations

import pandas as pd


def profile_table(df: pd.DataFrame) -> dict:
    """返回表格基础画像字典；n_rows 是测量行数，不等于独立实验单位数。"""
    columns = []
    for col in df.columns:
        series = df[col]
        columns.append(
            {
                "name": str(col),
                "dtype": str(series.dtype),
                "non_null": int(series.notna().sum()),
                "missing": int(series.isna().sum()),
                "missing_rate": round(float(series.isna().mean()), 4),
                "n_unique": int(series.nunique(dropna=True)),
            }
        )
    return {
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "columns": columns,
    }


def count_units(df: pd.DataFrame, unit_col: str) -> int:
    """独立实验单位计数（unit_id 去重）；unit_col 缺失时抛 KeyError。"""
    if unit_col not in df.columns:
        raise KeyError(f"列 {unit_col!r} 不存在，无法统计独立实验单位数。")
    return int(df[unit_col].nunique(dropna=True))
