"""运行存储：RunManifest 证据链落盘（V3 方案 §4.3）."""
from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from sciverify.config import LOG_DIR


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_of(obj: Any) -> str:
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


class RunManifest(BaseModel):
    """单次工作流运行的证据清单。"""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    mode: str  # LIVE | REPLAY_OFFLINE
    dataset_hash: str
    model_id: str
    plan_hash: str
    tool_call_ids: list[str] = []
    result_hash: str
    verification_status: str
    artifact_paths: list[str] = []
    created_at: str


class RunStore:
    """把 manifest 与事件流写到 logs/runs/<run_id>/ 下。"""

    def __init__(self, base_dir: str | Path | None = None) -> None:
        self.base = Path(base_dir) if base_dir else LOG_DIR / "runs"
        self.base.mkdir(parents=True, exist_ok=True)

    def new_run_id(self) -> str:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        return f"run_{ts}_{uuid.uuid4().hex[:6]}"

    def save(
        self,
        manifest: RunManifest,
        events: list[dict[str, Any]],
    ) -> Path:
        run_dir = self.base / manifest.run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = run_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest.model_dump(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        events_path = run_dir / "events.jsonl"
        with events_path.open("w", encoding="utf-8") as fh:
            for ev in events:
                fh.write(json.dumps(ev, ensure_ascii=False, default=str) + "\n")
        return manifest_path

    def load(self, run_id: str) -> RunManifest:
        p = self.base / run_id / "manifest.json"
        return RunManifest.model_validate_json(p.read_text(encoding="utf-8"))
