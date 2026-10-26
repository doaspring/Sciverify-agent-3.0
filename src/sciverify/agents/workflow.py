"""显式工作流状态机（V3 方案 §4.1 允许的"LangGraph 或等效显式状态机"）.

DataProfile → DesignReview → Plan(Intern-S2) → ToolSelection →
DeterministicExecution → Verification → (Report M5) ，含用户澄清分支。

LLM 边界（最多两处）：
  1) node_plan：Intern-S2 做自然语言任务理解 + 白名单工具选择（LIVE）；
  2) M5 的报告解释（受约束模板，不产生数字）。
其余节点全部是确定性代码；无 Key 时 PLAN 节点回退到确定性规则规划器并
显式标记 mode=REPLAY_OFFLINE，绝不伪装 LIVE。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from sciverify.api.budget import BudgetExceededError
from sciverify.api.intern_client import (
    InternApiError,
    InternClient,
    LiveApiBlockedError,
    extract_tool_calls,
)
from sciverify.config import MAX_AGENT_STEPS, MAX_REPLANS, MODEL, has_live_api
from sciverify.data.profile import profile_table
from sciverify.design.rules import Resolution, Severity, review_design
from sciverify.design.schema import DesignContext, DesignType
from sciverify.storage.run_store import RunManifest, RunStore, sha256_of
from sciverify.tools import stats as st
from sciverify.tools.registry import default_registry
from sciverify.verification import numeric as ver

MAX_PLANNER_RETRIES = 1  # 规划失败后的重规划上限（总尝试 = 1 + 此值）


@dataclass
class WorkflowState:
    """状态机的可序列化状态（事件流同时记录每个节点进入/退出）。"""

    df: pd.DataFrame
    design: DesignContext
    question: str
    mode: str = "REPLAY_OFFLINE"
    events: list[dict[str, Any]] = field(default_factory=list)

    profile: dict[str, Any] | None = None
    review: Any = None
    plan: dict[str, Any] | None = None
    plan_source: str = ""  # intern_s2 | rule_based_replay
    unit_df: pd.DataFrame | None = None
    tool_result: dict[str, Any] | None = None
    verification: dict[str, Any] | None = None
    status: str = "RUNNING"  # RUNNING | PASS | WARN_PASS | BLOCKED | FAILED
    blocked_reason: str = ""
    tool_call_ids: list[str] = field(default_factory=list)
    run_id: str = ""

    def log(self, node: str, status: str, **detail: Any) -> None:
        self.events.append(
            {
                "ts": time.time(),
                "node": node,
                "status": status,
                **detail,
            }
        )


# ---------------------------------------------------------------- 节点


def node_profile(state: WorkflowState) -> None:
    state.profile = profile_table(state.df)
    state.log(
        "DataProfile",
        "OK",
        n_rows=state.profile["n_rows"],
        n_columns=state.profile["n_columns"],
    )


def node_design_review(state: WorkflowState) -> None:
    state.review = review_design(state.df, state.design)
    overall = state.review.overall.value
    state.log(
        "DesignReview",
        overall,
        findings=[
            {"rule": f.rule_id, "severity": f.severity.value,
             "resolved": f.resolution_status == Resolution.RESOLVED}
            for f in state.review.findings
        ],
        n_rows=state.review.n_rows,
        n_units=state.review.n_units,
    )
    if overall == Severity.BLOCK.value:
        state.status = "BLOCKED"
        blockers = "; ".join(
            f"{f.rule_id}: {f.explanation}" for f in state.review.open_blockers
        )
        state.blocked_reason = blockers or "设计审查存在未解除的阻断"


def _design_summary(state: WorkflowState) -> str:
    d = state.design
    return (
        f"研究设计：分组列={d.group_column}，结局列={d.outcome_column}，"
        f"独立单位列={d.independent_unit_column}，"
        f"设计类型={d.design_type.value}，"
        f"技术重复列={d.technical_repeat_column or '无'}，"
        f"聚合策略={d.aggregation_policy.value}，"
        f"独立单位已确认={d.independence_confirmed}；"
        f"数据 n_rows={state.review.n_rows}，n_units={state.review.n_units}。"
    )


def _rule_based_plan(state: WorkflowState) -> dict[str, Any]:
    """确定性回退规划器（REPLAY 模式；也可作为 LIVE 失败后的最后回退，
    但回退时 plan_source 会如实标注 rule_based_replay，不冒充模型规划）。"""
    d = state.design
    if d.design_type == DesignType.PAIRED:
        return {
            "tool_name": "paired_t_test",
            "arguments": {
                "group_col": d.group_column,
                "value_col": d.outcome_column,
                "pair_col": d.pair_id_column or "",
            },
        }
    return {
        "tool_name": "welch_t_test",
        "arguments": {
            "group_col": d.group_column,
            "value_col": d.outcome_column,
        },
    }


def _validate_plan(plan: dict[str, Any], registry) -> tuple[bool, str]:
    name = plan.get("tool_name")
    if name not in registry.names:
        return False, f"工具 {name!r} 不在白名单 {registry.names}"
    args = plan.get("arguments")
    if not isinstance(args, dict):
        return False, "arguments 必须是 JSON 对象"
    return True, ""


def node_plan(state: WorkflowState, client: InternClient | None) -> None:
    """Intern-S2 规划（LIVE）：任务理解 + 白名单工具选择；REPLAY 回退。"""
    registry = default_registry()
    state.unit_df = st.prepare_analysis_frame(state.df, state.design)

    if state.mode == "LIVE":
        assert client is not None
        system = (
            "你是 SciVerify Agent 的分析规划器。你的唯一任务：根据用户的科研分析"
            "需求与研究设计摘要，从白名单工具中选择恰好一个工具并给出参数。"
            "禁止编造数值；禁止给出白名单之外的工具名；参数只用提供的列名。"
        )
        user = (
            f"用户问题：{state.question}\n研究设计摘要：{_design_summary(state)}\n"
            f"可用工具：{registry.names}\n"
            "请通过 tool_calls 返回你要调用的工具与参数。"
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        for attempt in range(1 + MAX_PLANNER_RETRIES):
            try:
                resp = client.chat_with_retry(
                    messages, tools=registry.schemas()
                )
            except BudgetExceededError as exc:
                # 公网预算耗尽：不重试、不冒充，显式降级并如实标注
                state.log("Plan", "BUDGET_EXCEEDED", error=str(exc)[:200])
                plan = _rule_based_plan(state)
                state.plan, state.plan_source = (
                    plan, "rule_based_replay(DAILY_BUDGET_EXCEEDED)"
                )
                state.log(
                    "Plan", "FALLBACK",
                    plan_source=state.plan_source, tool=plan["tool_name"],
                )
                return
            except (InternApiError, LiveApiBlockedError) as exc:
                state.log("Plan", "API_ERROR", error=str(exc)[:300], attempt=attempt)
                # LIVE 失败不允许伪装成功：显式回退并标注来源
                plan = _rule_based_plan(state)
                state.plan, state.plan_source = plan, "rule_based_replay(LIVE_API_ERROR)"
                state.log(
                    "Plan", "FALLBACK",
                    plan_source=state.plan_source, tool=plan["tool_name"],
                )
                return
            tool_calls = extract_tool_calls(resp["choices"][0]["message"])
            if tool_calls:
                call = tool_calls[0]
                state.tool_call_ids.append(call.get("id", ""))
                raw_args = call["function"].get("arguments", "{}")
                try:
                    args = raw_args if isinstance(raw_args, dict) else __import__(
                        "json"
                    ).loads(raw_args)
                except Exception:
                    args = None
                plan = {"tool_name": call["function"]["name"], "arguments": args}
                ok, why = _validate_plan(plan, registry)
                if ok:
                    state.plan, state.plan_source = plan, "intern_s2"
                    state.log(
                        "Plan", "OK", plan_source="intern_s2",
                        tool=plan["tool_name"], attempt=attempt,
                    )
                    return
                state.log("Plan", "INVALID", reason=why, attempt=attempt)
                messages.append({"role": "assistant", "content": "", "tool_calls": tool_calls})
                messages.append(
                    {
                        "role": "user",
                        "content": f"你给出的计划无效（{why}）。请重新只通过 "
                        "tool_calls 返回一个白名单工具及合法参数。",
                    }
                )
            else:
                state.log("Plan", "NO_TOOL_CALLS", attempt=attempt)
                messages.append(
                    {
                        "role": "user",
                        "content": "请直接通过 tool_calls 调用一个白名单工具，"
                        "不要文字作答。",
                    }
                )
        # 重规划耗尽 → BLOCKED（不静默回退，保持 LIVE 演示真实性）
        state.status = "BLOCKED"
        state.blocked_reason = (
            f"Intern-S2 规划连续 {1 + MAX_PLANNER_RETRIES} 次未给出合法白名单工具调用"
        )
        state.log("Plan", "EXHAUSTED")
        return

    # REPLAY_OFFLINE：确定性规则规划
    plan = _rule_based_plan(state)
    state.plan, state.plan_source = plan, "rule_based_replay"
    state.log("Plan", "OK", plan_source="rule_based_replay", tool=plan["tool_name"])


def node_execute(state: WorkflowState) -> None:
    registry = default_registry()
    if state.status == "BLOCKED" or state.plan is None:
        state.log("DeterministicExecution", "SKIPPED", reason="plan unavailable")
        return
    try:
        result = registry.execute(
            state.plan["tool_name"],
            state.plan["arguments"],
            state.unit_df,
            design=state.design,
        )
    except Exception as exc:
        state.status = "BLOCKED"
        state.blocked_reason = f"工具执行被拒绝：{exc}"
        state.log("DeterministicExecution", "REJECTED", error=str(exc)[:300])
        return
    state.tool_result = result
    state.log(
        "DeterministicExecution",
        "OK",
        tool=state.plan["tool_name"],
        n_units_used=result.get("n_units_used"),
        p_value=result.get("p_value"),
    )


def node_verify(state: WorkflowState) -> None:
    if state.status == "BLOCKED" or state.tool_result is None:
        state.log("Verification", "SKIPPED")
        return
    result = state.tool_result
    tool = result["tool"]
    unit_df, d = state.unit_df, state.design
    gcol, vcol = d.group_column, d.outcome_column

    try:
        if tool == "welch_t_test":
            samples = {
                str(g): grp[vcol].dropna().tolist()
                for g, grp in unit_df.groupby(unit_df[gcol].astype(str))
            }
            verdict = ver.verify_welch(samples, result)
        elif tool == "paired_t_test":
            wide = unit_df.pivot_table(
                index=d.pair_id_column, columns=gcol, values=vcol, aggfunc="first"
            ).sort_index()
            groups = sorted(wide.columns)
            diffs = (wide[groups[1]] - wide[groups[0]]).tolist()
            verdict = ver.verify_paired(diffs, result)
        elif tool == "describe_groups":
            values = {
                str(g): grp[vcol].dropna().tolist()
                for g, grp in unit_df.groupby(unit_df[gcol].astype(str))
            }
            verdict = ver.verify_describe(values, result)
        else:
            verdict = {"status": "SKIP", "checks": {"reason": "未知工具，无核验器"}}
    except Exception as exc:
        verdict = {"status": "FAIL", "checks": {"error": str(exc)[:300]}}

    state.verification = verdict
    state.log("Verification", verdict["status"], checks=verdict["checks"])
    if verdict["status"] == "FAIL":
        state.status = "BLOCKED"
        state.blocked_reason = "独立数值核验失败：工具结果与第二实现不一致，禁止出报告"
    elif state.status == "RUNNING":
        state.status = "PASS" if not _has_open_warn(state) else "WARN_PASS"


def _has_open_warn(state: WorkflowState) -> bool:
    return any(
        f.severity == Severity.WARN and f.resolution_status == Resolution.OPEN
        for f in state.review.findings
    )


# ---------------------------------------------------------------- 状态机入口


def run_workflow(
    df: pd.DataFrame,
    design: DesignContext,
    question: str = "请比较两组结局指标是否有差异",
    mode: str | None = None,
    client: InternClient | None = None,
    store: RunStore | None = None,
) -> WorkflowState:
    """执行完整状态机；返回带事件流的终态（报告节点 M5 接入）。"""
    if mode is None:
        mode = "LIVE" if has_live_api() else "REPLAY_OFFLINE"
    if mode == "LIVE" and not has_live_api():
        mode = "REPLAY_OFFLINE"  # 无 Key 绝不伪装 LIVE

    state = WorkflowState(df=df, design=design, question=question, mode=mode)
    store = store or RunStore()
    state.run_id = store.new_run_id()
    state.log("WorkflowStart", "OK", mode=mode, model_id=MODEL if mode == "LIVE" else "none")

    for _step in range(MAX_AGENT_STEPS):
        node_profile(state)
        node_design_review(state)
        if state.status == "BLOCKED":
            break
        node_plan(state, client)
        if state.status == "BLOCKED":
            break
        node_execute(state)
        node_verify(state)
        break  # 单轮计划-执行-核验；多轮规划在 M5 报告流程中扩展

    if state.status == "RUNNING":
        state.status = "FAILED"
        state.log("WorkflowEnd", "FAILED")

    # 证据链落盘
    manifest = RunManifest(
        run_id=state.run_id,
        mode=state.mode,
        dataset_hash=sha256_of({"n_rows": int(len(df))}),  # M5 换为文件 SHA256
        model_id=MODEL if state.mode == "LIVE" else "none(rule_based_replay)",
        plan_hash=sha256_of(state.plan) if state.plan else "",
        tool_call_ids=state.tool_call_ids,
        result_hash=sha256_of(state.tool_result) if state.tool_result else "",
        verification_status=(state.verification or {}).get("status", "SKIPPED"),
        artifact_paths=[],
        created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )
    path = store.save(manifest, state.events)
    state.log("WorkflowEnd", state.status, manifest=str(path))
    return state
