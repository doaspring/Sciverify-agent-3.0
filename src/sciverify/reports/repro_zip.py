"""可复现 ZIP：把一次运行的全部证据打包（数据+设计+manifest+事件+报告+说明）."""
from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

from sciverify.storage.run_store import sha256_of

REPRO_README = """# 复现说明（run_id={run_id}）

本 ZIP 是 SciVerify Agent 一次分析的完整复现包，内容：

| 文件 | 说明 |
|---|---|
| original_data | 上传的原始数据文件（字节原样） |
| design.json | 用户确认的研究设计（DesignContext 快照） |
| manifest.json | 运行清单（模式、工具调用ID、结果哈希、核验状态） |
| events.jsonl | 状态机逐节点事件流 |
| tool_result.json | 统计工具结构化输出（报告数字的唯一来源） |
| report.html | 最终 HTML 报告（自包含，可直接打开） |

## 复现步骤

```bash
# 1. 环境：Python 3.12 + pip install -r requirements.lock + pip install -e . --no-deps
# 2. 校验原始数据哈希（应等于 manifest 中 dataset_hash 对应文件的实际 SHA256）：
python -c "import hashlib;print(hashlib.sha256(open('original_data','rb').read()).hexdigest())"
# 3. 复跑（离线确定性路径，数值应与 tool_result.json 完全一致）：
python scripts/run_workflow.py --replay
```

注意：REPLAY_OFFLINE 模式为确定性计算，同数据同设计必然得到相同数值；
LIVE 模式的模型规划文本可能措辞不同，但工具、参数与数值仍由本包数据决定。
生成时间：{created_at}
"""


def build_repro_zip(
    out_path: str | Path,
    *,
    run_id: str,
    data_path: str | Path,
    design_snapshot: dict[str, Any],
    manifest: dict[str, Any],
    events: list[dict[str, Any]],
    tool_result: dict[str, Any] | None,
    report_html: str,
    created_at: str,
) -> Path:
    """打包复现 ZIP 并返回路径；report_html 为完整 HTML 字符串。"""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    data = Path(data_path)

    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(data, arcname=f"original_data{data.suffix}")
        zf.writestr(
            "design.json",
            json.dumps(design_snapshot, ensure_ascii=False, indent=2),
        )
        zf.writestr(
            "manifest.json",
            json.dumps(manifest, ensure_ascii=False, indent=2),
        )
        zf.writestr(
            "events.jsonl",
            "\n".join(json.dumps(e, ensure_ascii=False, default=str) for e in events),
        )
        if tool_result:
            zf.writestr(
                "tool_result.json",
                json.dumps(tool_result, ensure_ascii=False, indent=2),
            )
        zf.writestr("report.html", report_html)
        zf.writestr(
            "README_repro.txt",
            REPRO_README.format(run_id=run_id, created_at=created_at),
        )
    return out
