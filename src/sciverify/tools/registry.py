"""受控工具注册表：模型只能通过白名单名称调用本地确定性函数.

规则：
- 注册表之外的一切"工具名"一律拒绝；
- 声明 requires_confirmed_units=True 的工具（推断性检验）在执行前
  必须通过 assert_inference_allowed 门控（M1 验收硬标准）；
- 参数一律 Pydantic extra=forbid 校验。
"""
from __future__ import annotations

import json
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict, ValidationError

from sciverify.design.confirm import assert_inference_allowed
from sciverify.tools import stats


class DescribeGroupsArgs(BaseModel):
    """describe_groups 的严格参数模型：多余键直接拒绝（extra=forbid）。"""

    model_config = ConfigDict(extra="forbid")

    group_col: str
    value_col: str


class WelchTTestArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    group_col: str
    value_col: str
    alpha: float = 0.05


class PairedTTestArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    group_col: str
    value_col: str
    pair_col: str
    alpha: float = 0.05


TOOLS_SCHEMA: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "describe_groups",
            "description": (
                "按实验分组对指定数值结局列计算描述统计"
                "（每组 n_rows/mean/std/min/max）。用于两组比较前的数据概览。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "group_col": {
                        "type": "string",
                        "description": "分组列名，例如 group",
                    },
                    "value_col": {
                        "type": "string",
                        "description": "数值结局列名，例如 enzyme_activity_u_ml",
                    },
                },
                "required": ["group_col", "value_col"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "welch_t_test",
            "description": (
                "独立两组 Welch t 检验（不假设方差齐性）。输入必须是"
                "独立实验单位级数据（每单位一行）。返回统计量、P 值、"
                "均值差及其置信区间。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "group_col": {"type": "string", "description": "分组列名"},
                    "value_col": {"type": "string", "description": "数值结局列名"},
                    "alpha": {
                        "type": "number",
                        "description": "显著性水平，默认 0.05",
                    },
                },
                "required": ["group_col", "value_col"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "paired_t_test",
            "description": (
                "配对 t 检验。必须提供配对标识列 pair_col，按 pair_id 严格匹配，"
                "禁止按行序配对。返回统计量、P 值、均值差及其置信区间。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "group_col": {"type": "string", "description": "条件列名（两水平）"},
                    "value_col": {"type": "string", "description": "数值结局列名"},
                    "pair_col": {"type": "string", "description": "配对标识列名"},
                    "alpha": {
                        "type": "number",
                        "description": "显著性水平，默认 0.05",
                    },
                },
                "required": ["group_col", "value_col", "pair_col"],
            },
        },
    }
]


def describe_groups_impl(df: Any, raw_args: dict[str, Any]) -> dict[str, Any]:
    args = DescribeGroupsArgs.model_validate(raw_args)
    return stats.describe_groups_result(df, args.group_col, args.value_col)


def welch_impl(df: Any, raw_args: dict[str, Any]) -> dict[str, Any]:
    args = WelchTTestArgs.model_validate(raw_args)
    result = stats.welch_t_test(df, args.group_col, args.value_col, args.alpha)
    return result.model_dump(mode="json")


def paired_impl(df: Any, raw_args: dict[str, Any]) -> dict[str, Any]:
    args = PairedTTestArgs.model_validate(raw_args)
    result = stats.paired_t_test(
        df, args.group_col, args.value_col, args.pair_col, args.alpha
    )
    return result.model_dump(mode="json")


class ToolRegistry:
    """白名单注册表：名称 -> (schema, 参数模型, 实现函数, 是否需确认设计)。"""

    def __init__(self) -> None:
        self._entries: dict[str, dict[str, Any]] = {}

    def register(
        self,
        name: str,
        schema: dict[str, Any],
        args_model: type[BaseModel],
        impl: Callable[[Any, dict[str, Any]], Any],
        requires_confirmed_units: bool = False,
    ) -> None:
        """注册工具；requires_confirmed_units=True 的工具执行前做推断门控。"""
        self._entries[name] = {
            "schema": schema,
            "args_model": args_model,
            "impl": impl,
            "requires_confirmed_units": requires_confirmed_units,
        }

    @property
    def names(self) -> list[str]:
        return sorted(self._entries)

    def schemas(self) -> list[dict[str, Any]]:
        return [e["schema"] for e in self._entries.values()]

    def execute(
        self,
        name: str,
        raw_args: dict[str, Any] | str,
        df: Any,
        design: Any = None,
    ) -> Any:
        """执行白名单工具；未知名称/非法参数/推断未确认一律拒绝。"""
        if name not in self._entries:
            raise ValueError(f"工具 {name!r} 不在白名单 {self.names} 中")
        entry = self._entries[name]
        if entry["requires_confirmed_units"]:
            assert_inference_allowed(design)  # 未确认 → InferenceBlockedError
        if isinstance(raw_args, str):
            try:
                raw_args = json.loads(raw_args)
            except json.JSONDecodeError as exc:
                raise ValueError(f"工具 {name!r} 参数不是合法 JSON：{exc}") from exc
        if not isinstance(raw_args, dict):
            raise ValueError(f"工具 {name!r} 参数必须是 JSON 对象")
        try:
            args = entry["args_model"].model_validate(raw_args)
        except ValidationError as exc:
            raise ValueError(f"工具 {name!r} 参数校验失败：{exc}") from exc
        return entry["impl"](df, args.model_dump())


def default_registry() -> ToolRegistry:
    """默认注册表：描述统计（免确认）+ Welch/配对 t（需设计确认门控）。"""
    reg = ToolRegistry()
    reg.register(
        "describe_groups",
        TOOLS_SCHEMA[0],
        DescribeGroupsArgs,
        describe_groups_impl,
        requires_confirmed_units=False,
    )
    reg.register(
        "welch_t_test",
        TOOLS_SCHEMA[1],
        WelchTTestArgs,
        welch_impl,
        requires_confirmed_units=True,
    )
    reg.register(
        "paired_t_test",
        TOOLS_SCHEMA[2],
        PairedTTestArgs,
        paired_impl,
        requires_confirmed_units=True,
    )
    return reg
