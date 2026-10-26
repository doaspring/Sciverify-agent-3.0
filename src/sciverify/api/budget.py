"""公网部署的每日 LIVE 调用预算（V3 方案 §4.6-3：限流+每日预算+文件限制）.

目的：公开体验链接意味着任何人都能触发真实书生 API 调用；本模块以
"每天最多 N 次真实调用"保护用户的 Key 配额。计数持久化到临时目录
（云端容器重启即清零，属可接受的保守策略）。
"""
from __future__ import annotations

import datetime as _dt
import json
import tempfile
from pathlib import Path

from sciverify.config import DAILY_API_BUDGET

_STATE_FILE = Path(tempfile.gettempdir()) / "sciverify_api_budget.json"


class BudgetExceededError(RuntimeError):
    """当日 LIVE 调用预算耗尽（公网保护，非系统故障）。"""


def _load_state() -> dict:
    today = _dt.date.today().isoformat()
    try:
        data = json.loads(_STATE_FILE.read_text(encoding="utf-8"))
        if data.get("date") == today:
            return data
    except Exception:
        pass
    return {"date": today, "count": 0}


def _save_state(state: dict) -> None:
    try:
        _STATE_FILE.write_text(json.dumps(state), encoding="utf-8")
    except Exception:
        pass  # 只读文件系统时退化为不持久化（每次重启重新计数）


def usage_today() -> tuple[int, int]:
    """返回 (今日已用次数, 每日上限)。"""
    s = _load_state()
    return int(s["count"]), int(DAILY_API_BUDGET)


def consume_budget() -> None:
    """发起真实 API 调用前扣减预算；超限抛 BudgetExceededError。"""
    s = _load_state()
    s["count"] = int(s["count"]) + 1
    _save_state(s)
    if int(s["count"]) > int(DAILY_API_BUDGET):
        raise BudgetExceededError(
            f"今日 LIVE 调用预算已用尽（上限 {DAILY_API_BUDGET} 次/天，公网保护限额）。"
            "仍可使用内置演示数据完成离线流程演示；预算每日自动恢复。"
        )
