"""Claim Guard：报告数字必须与 ToolResult 完全一致（V3 §4.5 铁律）.

模板为每个数值字段输出 data-field/data-value 属性；本模块解析 HTML
并逐字段与 ToolResult 比对，任何不一致 → FAIL（此时禁止发布报告）。
"""
from __future__ import annotations

import json
import re
from html import unescape as _html_unescape
from typing import Any

_DATA_ATTR_RE = re.compile(
    r'data-field="(?P<field>[^"]+)"(?:\s+data-value=(?P<q>[\'"])(?P<value>.*?)(?P=q))?'
)

_NUMERIC_FIELDS = {
    "n_units_used": lambda r: r.get("n_units_used"),
    "statistic": lambda r: r.get("statistic"),
    "p_value": lambda r: r.get("p_value"),
    "mean_difference": lambda r: r.get("mean_difference"),
}
_JSON_FIELDS = {
    "n_by_group": lambda r: r.get("n_by_group"),
    "mean_by_group": lambda r: r.get("mean_by_group"),
    "confidence_interval": lambda r: list(r.get("confidence_interval") or []),
}


def extract_report_numbers(html: str) -> dict[str, str]:
    """提取 HTML 中带 data-field 标注的数值（字符串形态，已反转义）。"""
    out: dict[str, str] = {}
    for m in _DATA_ATTR_RE.finditer(html):
        field = m.group("field")
        value = m.group("value")
        if value is not None and field not in out:
            out[field] = _html_unescape(value)
    return out


def check_report_against_result(html: str, tool_result: dict[str, Any]) -> dict[str, Any]:
    """核对报告数值与 ToolResult；返回 {status, mismatches, checked}。"""
    if not tool_result:
        return {"status": "FAIL", "mismatches": ["报告含统计结果但 ToolResult 缺失"],
                "checked": 0}
    reported = extract_report_numbers(html)
    mismatches: list[str] = []
    checked = 0

    for field, getter in _NUMERIC_FIELDS.items():
        expected = getter(tool_result)
        if expected is None:
            continue
        if field not in reported:
            mismatches.append(f"报告缺少字段 {field}")
            continue
        checked += 1
        if abs(float(reported[field]) - float(expected)) > 1e-12 * max(
            1.0, abs(float(expected))
        ):
            mismatches.append(
                f"{field}: 报告={reported[field]} ≠ 工具={expected}"
            )

    for field, getter in _JSON_FIELDS.items():
        expected = getter(tool_result)
        if expected is None:
            continue
        if field not in reported:
            mismatches.append(f"报告缺少字段 {field}")
            continue
        checked += 1
        try:
            reported_val = json.loads(reported[field])
        except json.JSONDecodeError:
            mismatches.append(f"{field}: 报告值不是合法 JSON")
            continue
        if reported_val != expected and _canonical(reported_val) != _canonical(expected):
            mismatches.append(f"{field}: 报告={reported_val} ≠ 工具={expected}")

    return {
        "status": "PASS" if not mismatches else "FAIL",
        "mismatches": mismatches,
        "checked": checked,
    }


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False)
