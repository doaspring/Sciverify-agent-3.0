"""数据摄取：CSV/XLSX 白名单加载、大小限制、SHA256 指纹.

约束（V3 方案 §3.1/§4.2）：仅接受 CSV/XLSX；上传大小受 MAX_FILE_MB 限制；
CSV 首行以 '#' 开始的注释行被忽略（用于 SYNTHETIC 数据标注）。
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from sciverify.config import MAX_FILE_MB

ALLOWED_SUFFIXES = {".csv", ".tsv", ".xlsx", ".json"}


class DataIngestError(Exception):
    """数据摄取失败基类。"""


class UnsupportedFormatError(DataIngestError):
    """文件格式不在白名单内。"""


class FileTooLargeError(DataIngestError):
    """文件超出大小限制。"""


class EmptyTableError(DataIngestError):
    """表格为空或没有数据行。"""


def file_sha256(path: str | Path) -> str:
    """计算文件 SHA256（证据链：原始数据指纹）。"""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def list_sheets(path: str | Path) -> list[str]:
    """枚举 XLSX 工作表名（其他格式返回占位符，方便 UI 统一处理）。"""
    p = Path(path)
    if p.suffix.lower() not in ALLOWED_SUFFIXES:
        raise UnsupportedFormatError(
            f"不支持的文件格式 {p.suffix!r}；仅接受 {sorted(ALLOWED_SUFFIXES)}。"
        )
    if p.suffix.lower() == ".xlsx":
        return list(pd.ExcelFile(p).sheet_names)
    suffix = p.suffix.lower().lstrip(".")
    return [f"<{suffix}>"]


def load_table(
    path: str | Path,
    max_mb: float | None = None,
    sheet_name: str | int = 0,
) -> pd.DataFrame:
    """加载 CSV/XLSX 为 DataFrame，强制执行格式与大小限制。

    Raises:
        UnsupportedFormatError / FileTooLargeError / EmptyTableError
    """
    p = Path(path)
    limit_mb = MAX_FILE_MB if max_mb is None else float(max_mb)

    if p.suffix.lower() not in ALLOWED_SUFFIXES:
        raise UnsupportedFormatError(
            f"不支持的文件格式 {p.suffix!r}；仅接受 {sorted(ALLOWED_SUFFIXES)}。"
        )
    if not p.exists():
        raise DataIngestError(f"文件不存在：{p}")
    size_mb = p.stat().st_size / (1024 * 1024)
    if size_mb > limit_mb:
        raise FileTooLargeError(
            f"文件 {size_mb:.2f} MB 超过限制 {limit_mb} MB。"
        )

    if p.suffix.lower() in {".csv", ".tsv"}:
        # '#' 开头的行是元数据注释（含 SYNTHETIC 标注），不是数据
        sep = "\t" if p.suffix.lower() == ".tsv" else ","
        df = pd.read_csv(p, sep=sep, comment="#")
    elif p.suffix.lower() == ".json":
        # JSON 数组记录（[{"col": val, ...}, ...]），无注释行概念
        df = pd.read_json(p, orient="records")
    else:
        df = pd.read_excel(p, sheet_name=sheet_name)

    if df.empty:
        raise EmptyTableError(f"表格没有数据行：{p.name}")
    return df
