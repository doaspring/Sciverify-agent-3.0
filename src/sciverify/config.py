"""集中配置：环境变量解析与项目路径.

真实 Key 只允许出现在本地 .env（已被 .gitignore 排除），绝不进入 Git、日志或报告。
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# sciverify-agent/ 项目根目录（src/sciverify/config.py 向上三级）
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 优先加载项目根 .env，再叠加当前目录 .env（后加载不覆盖已存在变量）
load_dotenv(PROJECT_ROOT / ".env")
load_dotenv(override=False)


def _env_bool(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes")


def _secret_or_env(name: str, default: str = "") -> str:
    """配置读取顺序：Streamlit Secrets（云端部署）→ 环境变量/.env（本地）。

    Streamlit Cloud 的 Secrets 在平台后台配置，等价于云端 .env；本地与
    测试环境没有 secrets 文件，自动落到环境变量路径。
    """
    try:
        import streamlit as st

        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass  # 非 Streamlit 运行时（pytest/脚本）或无 secrets 文件
    return os.getenv(name, default)


API_KEY: str = _secret_or_env("INTERN_API_KEY").strip()
MODEL: str = _secret_or_env("INTERN_MODEL", "intern-s2").strip()
API_BASE_URL: str = _secret_or_env(
    "INTERN_API_BASE_URL", "https://chat.intern-ai.org.cn/api/v1"
).rstrip("/")
THINKING_MODE: bool = _env_bool("INTERN_THINKING_MODE", "true")
API_TIMEOUT_S: float = float(os.getenv("INTERN_TIMEOUT_S", "120"))

MOCK_MODE: bool = _env_bool("MOCK_MODE", "false")
DAILY_API_BUDGET: int = int(_secret_or_env("DAILY_API_BUDGET", "300"))
MAX_FILE_MB: float = float(os.getenv("MAX_FILE_MB", "15"))
MAX_AGENT_STEPS: int = int(os.getenv("MAX_AGENT_STEPS", "8"))
MAX_REPLANS: int = int(os.getenv("MAX_REPLANS", "2"))

LOG_DIR: Path = Path(os.getenv("SCIVERIFY_LOG_DIR", str(PROJECT_ROOT / "logs")))
LOG_DIR.mkdir(parents=True, exist_ok=True)

DEMOS_DIR: Path = PROJECT_ROOT / "demos"


def has_live_api() -> bool:
    """是否具备真实书生 API 调用条件（有 Key 且未强制 Mock）。"""
    return bool(API_KEY) and not MOCK_MODE


def live_status_label() -> str:
    """UI 状态标签：必须如实反映运行模式，禁止伪装 LIVE。"""
    if has_live_api():
        return "LIVE Intern-S2"
    if MOCK_MODE:
        return "MOCK MODE（仅离线开发）"
    return "LIVE_API_UNAVAILABLE（缺少 INTERN_API_KEY）"
