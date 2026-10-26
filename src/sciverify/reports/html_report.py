"""HTML 报告生成：数值 100% 来自 ToolResult 的确定性 Jinja2 渲染.

规则（V3 方案 §3.5）：
- 模板只做展示，不做任何算术；每个数值带 data-field/data-value 属性，
  供 claim_guard 反向核对"报告数字 == 工具输出"；
- 报告必须包含：run_id、数据 SHA256、设计确认快照、审查发现、分析计划、
  ToolResult、核验结论、库版本、模式标签（LIVE/REPLAY）。
"""
from __future__ import annotations

import base64
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jinja2 import Environment, StrictUndefined, select_autoescape

from sciverify.storage.run_store import sha256_of

_TEMPLATE = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>SciVerify 分析报告 {{ run_id }}</title>
<style>
  body { font-family: "Segoe UI", "Microsoft YaHei", sans-serif; margin: 2rem auto;
         max-width: 900px; color: #1c2b26; line-height: 1.6; }
  h1 { color: #1f6f54; border-bottom: 3px solid #1f6f54; padding-bottom: .3rem; }
  h2 { color: #1f6f54; margin-top: 1.6rem; }
  table { border-collapse: collapse; width: 100%; margin: .6rem 0; }
  th, td { border: 1px solid #c9d6d0; padding: .45rem .6rem; text-align: left; }
  th { background: #eef5f1; }
  .badge-pass { background: #d9f0e2; color: #14532d; padding: .15rem .6rem;
                border-radius: 1rem; font-weight: 600; }
  .badge-warn { background: #fdf3d7; color: #713f12; padding: .15rem .6rem;
                border-radius: 1rem; font-weight: 600; }
  .badge-block { background: #fde2e2; color: #7f1d1d; padding: .15rem .6rem;
                 border-radius: 1rem; font-weight: 600; }
  .muted { color: #5c6b64; font-size: .9rem; }
  footer { margin-top: 2rem; border-top: 1px solid #c9d6d0; padding-top: .6rem;
           font-size: .85rem; color: #5c6b64; }
</style>
</head>
<body>
<h1>SciVerify Agent 科研统计分析报告</h1>
<p>
  <span class="muted">运行</span> <code>{{ run_id }}</code>
  &nbsp;|&nbsp; <span class="muted">模式</span>
  {% if mode == "LIVE" %}<strong>LIVE Intern-S2（书生真实 API）</strong>
  {% else %}<strong>REPLAY_OFFLINE（确定性规则规划，非模型调用）</strong>{% endif %}
  &nbsp;|&nbsp; <span class="muted">生成时间</span> {{ created_at }}
</p>

<h2>1. 数据与设计确认</h2>
<table>
  <tr><th>项</th><th>值</th></tr>
  <tr><td>数据文件</td><td><code>{{ dataset_name }}</code></td></tr>
  <tr><td>原始数据 SHA256</td><td><code>{{ dataset_sha256 }}</code></td></tr>
  <tr><td>测量行数 n_rows</td><td data-field="n_rows">{{ n_rows }}</td></tr>
  <tr><td>独立实验单位数 n_units</td><td data-field="n_units">{{ n_units }}</td></tr>
  <tr><td>分组列 / 结局列 / 单位列</td>
      <td>{{ design.group_column }} / {{ design.outcome_column }} /
          {{ design.independent_unit_column }}</td></tr>
  <tr><td>设计类型</td><td>{{ design.design_type }}</td></tr>
  <tr><td>技术重复列 / 聚合策略</td>
      <td>{{ design.technical_repeat_column or "无" }} /
          {{ design.aggregation_policy }}</td></tr>
  <tr><td>单位语义（用户确认）</td><td>{{ design.unit_semantics or "（未确认）" }}</td></tr>
  <tr><td>确认时间戳</td><td>{{ design.confirmation_timestamp or "—" }}</td></tr>
</table>

<h2>2. 设计风险审查</h2>
<p>结论：<span class="badge-{{ review_badge_class }}">{{ review_overall }}</span>
   （n_rows={{ n_rows }}，n_units={{ n_units }}，各组单位数 {{ n_units_by_group }}）</p>
{% if findings %}
<table>
  <tr><th>规则</th><th>级别</th><th>状态</th><th>说明</th><th>所需动作</th></tr>
  {% for f in findings %}
  <tr>
    <td><code>{{ f.rule_id }}</code></td>
    <td>{{ f.severity }}</td>
    <td>{{ f.resolution_status }}</td>
    <td>{{ f.explanation }}</td>
    <td>{{ f.required_action }}</td>
  </tr>
  {% endfor %}
</table>
{% else %}
<p>六条规则全部无触发。</p>
{% endif %}

{% if tool_result %}
<h2>3. 分析计划与执行</h2>
<p>工具选择：<strong><code>{{ plan.tool_name }}</code></strong>
   <span class="muted">（规划来源：{{ plan_source }}）</span><br>
   参数：<code>{{ args_json }}</code></p>

<h2>4. 统计结果（全部来自本地确定性计算，未经模型生成）</h2>
<table>
  <tr><th>字段</th><th>值</th></tr>
  <tr><td>工具</td><td data-field="tool">{{ tool_result.tool }}</td></tr>
  <tr><td>有效样本量 n_units</td>
      <td data-field="n_units_used" data-value="{{ tool_result.n_units_used }}">
          {{ tool_result.n_units_used }}</td></tr>
  <tr><td>各组单位数</td><td data-field="n_by_group" data-value='{{ n_by_group_json }}'>
      {{ tool_result.n_by_group }}</td></tr>
  <tr><td>各组均值</td><td data-field="mean_by_group" data-value='{{ mean_by_group_json }}'>
      {{ mean_by_group_str }}</td></tr>
  <tr><td>统计量</td>
      <td data-field="statistic" data-value="{{ tool_result.statistic }}">
          {{ "%.4f"|format(tool_result.statistic) }}</td></tr>
  <tr><td>P 值（双侧）</td>
      <td data-field="p_value" data-value="{{ tool_result.p_value }}">
          {{ "%.6g"|format(tool_result.p_value) }}</td></tr>
  <tr><td>均值差</td>
      <td data-field="mean_difference" data-value="{{ tool_result.mean_difference }}">
          {{ "%.4f"|format(tool_result.mean_difference) }}</td></tr>
  <tr><td>95% CI</td>
      <td data-field="confidence_interval"
          data-value="{{ ci_json }}">
          [{{ "%.4f"|format(tool_result.confidence_interval[0]) }},
           {{ "%.4f"|format(tool_result.confidence_interval[1]) }}]</td></tr>
  <tr><td>差值定义</td><td>{{ tool_result.difference_definition }}</td></tr>
</table>
{% if tool_result.warnings %}
<ul>{% for w in tool_result.warnings %}<li>⚠️ {{ w }}</li>{% endfor %}</ul>
{% endif %}
<p class="muted">计算库版本：{{ lib_versions_json }}</p>
{% if boxplot_png_b64 %}
<h2>5. 图表</h2>
<img alt="按实验单位的分组箱线图" style="max-width:520px"
     src="data:image/png;base64,{{ boxplot_png_b64 }}">
{% endif %}

<h2>6. 独立数值核验</h2>
<p>结论：<span class="badge-{% if verification.status == "PASS" %}pass{% else %}block{% endif %}">
   {{ verification.status }}</span>
   —— 第二实现（math-only，含独立不完全 Beta 算法）逐字段复算：
   {{ verification.checks_json }}</p>
{% else %}
<h2>3. 分析计划与执行</h2>
<p><span class="badge-block">BLOCKED</span> 本次运行被设计审查阻断，未产生任何统计结果。
{{ blocked_reason }}</p>
{% endif %}

<footer>
  SciVerify Agent 3.0 · 证据链：run_id={{ run_id }} ·
  result_hash={{ result_hash or "—（无结果）" }} ·
  plan_hash={{ plan_hash or "—" }} ·
  本报告所有数值由 Jinja2 模板从 ToolResult 结构化字段直接填充，可由
  claim_guard 自动核对。
</footer>
</body>
</html>
"""


def _badge_class(overall: str) -> str:
    return {"PASS": "pass", "WARN": "warn", "BLOCK": "block"}.get(overall, "block")


def render_report_html(
    *,
    run_id: str,
    mode: str,
    dataset_name: str,
    dataset_sha256: str,
    n_rows: int,
    n_units: int,
    design: dict[str, Any],
    review_overall: str,
    n_units_by_group: dict[str, int],
    findings: list[dict[str, Any]],
    plan: dict[str, Any] | None,
    plan_source: str,
    tool_result: dict[str, Any] | None,
    verification: dict[str, Any] | None,
    boxplot_png_path: str | Path | None = None,
    blocked_reason: str = "",
    result_hash: str = "",
    plan_hash: str = "",
) -> str:
    """渲染 HTML 报告；数值字段全部来自传入的结构化对象。"""
    env = Environment(
        autoescape=select_autoescape(["html"]), undefined=StrictUndefined
    )
    tpl = env.from_string(_TEMPLATE)

    png_b64 = ""
    if boxplot_png_path and Path(boxplot_png_path).exists():
        png_b64 = base64.b64encode(Path(boxplot_png_path).read_bytes()).decode("ascii")

    verification = verification or {"status": "SKIPPED", "checks": {}}
    tr = tool_result or {}

    return tpl.render(
        run_id=run_id,
        mode=mode,
        created_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        dataset_name=dataset_name,
        dataset_sha256=dataset_sha256,
        n_rows=n_rows,
        n_units=n_units,
        design=design,
        review_overall=review_overall,
        review_badge_class=_badge_class(review_overall),
        n_units_by_group=n_units_by_group,
        findings=findings,
        plan=plan or {},
        plan_source=plan_source,
        args_json=json.dumps((plan or {}).get("arguments", {}), ensure_ascii=False),
        tool_result=tr or None,
        n_by_group_json=json.dumps(tr.get("n_by_group", {}), sort_keys=True),
        mean_by_group_json=json.dumps(tr.get("mean_by_group", {}), sort_keys=True),
        mean_by_group_str={
            k: round(v, 4) for k, v in tr.get("mean_by_group", {}).items()
        },
        ci_json=json.dumps(list(tr.get("confidence_interval") or [])),
        lib_versions_json=json.dumps(tr.get("library_versions", {}), ensure_ascii=False),
        boxplot_png_b64=png_b64,
        verification={**verification,
                     "checks_json": json.dumps(verification.get("checks", {}),
                                               ensure_ascii=False)},
        blocked_reason=blocked_reason,
        result_hash=result_hash,
        plan_hash=plan_hash,
    )


def save_report(html: str, out_path: str | Path) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out


def build_report_for_state(state, dataset_name: str, dataset_sha256: str,
                           boxplot_png_path: str | Path | None = None) -> str:
    """从 WorkflowState 组装报告所需的全部输入（数值零手工）。"""
    return render_report_html(
        run_id=state.run_id,
        mode=state.mode,
        dataset_name=dataset_name,
        dataset_sha256=dataset_sha256,
        n_rows=state.review.n_rows,
        n_units=state.review.n_units,
        design=state.design.snapshot(),
        review_overall=state.review.overall.value,
        n_units_by_group=state.review.n_units_by_group,
        findings=[f.model_dump(mode="json") for f in state.review.findings],
        plan=state.plan,
        plan_source=state.plan_source,
        tool_result=state.tool_result,
        verification=state.verification,
        boxplot_png_path=boxplot_png_path,
        blocked_reason=state.blocked_reason,
        result_hash=sha256_of(state.tool_result) if state.tool_result else "",
        plan_hash=sha256_of(state.plan) if state.plan else "",
    )
