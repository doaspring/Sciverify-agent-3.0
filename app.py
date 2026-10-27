"""SciVerify Agent 3.0 — Streamlit 应用（M1：数据入口 + 设计确认）.

界面区块（V3 方案 §4.1）：①数据入口 → ②数据画像 → ③设计确认
→ ④智能体执行（M4）→ ⑤统计结果（M3）→ ⑥证据导出（M5）。
状态标签必须如实反映 LIVE/MOCK/UNAVAILABLE，禁止伪装。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sciverify.config import (  # noqa: E402
    DEMOS_DIR,
    LOG_DIR,
    MODEL,
    has_live_api,
    live_status_label,
)
from sciverify.data.ingest import file_sha256, list_sheets, load_table  # noqa: E402
from sciverify.data.profile import count_units, profile_table  # noqa: E402
from sciverify.design import (  # noqa: E402
    QUESTION_UNIT_MEANING,
    DesignContext,
    DesignType,
    Resolution,
    ReviewReport,
    Severity,
    now_utc,
    review_design,
)

st.set_page_config(page_title="SciVerify Agent 3.0", page_icon="🧪", layout="wide")

st.title("SciVerify Agent 3.0 — 科研统计设计审查与可信结论验证")
st.caption("书生 Intern-S2 驱动 · 实验单位防错 · 确定性统计计算 · 证据链可复现")

# —— 运行模式标识（如实显示）——
label = live_status_label()
if has_live_api():
    st.sidebar.success(f"运行模式：{label}")
else:
    st.sidebar.warning(f"运行模式：{label}")
st.sidebar.caption(f"模型：{MODEL}（非流式 · thinking_mode）")
st.sidebar.divider()
st.sidebar.markdown(
    "**隐私提示**：演示数据均为合成（SYNTHETIC）；请勿上传未授权的真实实验数据。"
)

# ============ ① 数据入口 ============
st.header("① 数据入口")
demo_files = {
    "案例A 正常：两独立组各6批次（12行）": "valid_independent.csv",
    "案例B 陷阱：4批次×3技术重复（24行）": "technical_replicates.csv",
    "案例C 配对陷阱：前后对比缺pair_id": "paired_missing_ids.csv",
    "案例D 演练：两构型各8批次·零阻断全流程": "practice_clean.csv",
    "案例F 大模型评测：两提示策略×8基准（零阻断）": "llm_eval_clean.csv",
    "案例G 大模型陷阱：同任务5次采样≠独立实验（40行）": "llm_sampling_replicates.csv",
    "案例H 安全评测：两漏洞扫描器×8样本集（零阻断）": "sec_scan_clean.csv",
    "案例I 安全陷阱：同样本集3次扫描≠3份证据（24行）": "sec_scan_replicates.csv",
    "案例E XLSX：双工作表（含README）": "xlsx_demo.xlsx",
}

if "df" not in st.session_state:
    st.session_state.df = None
    st.session_state.df_name = None
    st.session_state.df_sha = None


def _reset_downstream_state() -> None:
    """数据更换时使下游全部失效：旧设计/审查/结果绝不能用于新数据."""
    for key in (
        "design", "review", "tool_result", "boxplot_path",
        "wf_state_events", "wf_status", "wf_reason",
        "report_path", "zip_path",
    ):
        st.session_state.pop(key, None)

col1, col2 = st.columns([3, 2])
with col1:
    choice = st.selectbox("一键载入合成演示数据", list(demo_files))
    if st.button("载入演示数据", type="primary"):
        path = DEMOS_DIR / demo_files[choice]
        sheet = 0
        if path.suffix.lower() == ".xlsx":
            sheets = list_sheets(path)
            data_sheets = [s for s in sheets if s.lower() != "readme"]
            sheet = data_sheets[0] if data_sheets else 0
        st.session_state.df = load_table(path, sheet_name=sheet)
        st.session_state.df_name = path.name
        st.session_state.df_sha = file_sha256(path)
        _reset_downstream_state()
with col2:
    upload = st.file_uploader(
        "或上传 CSV/TSV/XLSX/JSON（≤15MB，需授权数据）",
        type=["csv", "tsv", "xlsx", "json"],
    )
    if upload is not None:
        tmp_dir = DEMOS_DIR.parent / "uploads"
        tmp_dir.mkdir(exist_ok=True)
        tmp = tmp_dir / f"upload_{upload.name}"
        tmp.write_bytes(upload.getvalue())
        try:
            sheet = 0
            if tmp.suffix.lower() == ".xlsx":
                sheets = list_sheets(tmp)
                data_sheets = [s for s in sheets if s.lower() != "readme"]
                if len(data_sheets) > 1:
                    sheet = st.selectbox("选择工作表", data_sheets, key=f"sheet_{upload.name}")
                else:
                    sheet = data_sheets[0]
            st.session_state.df = load_table(tmp, sheet_name=sheet)
            st.session_state.df_name = upload.name
            st.session_state.df_sha = file_sha256(tmp)
            _reset_downstream_state()
        except Exception as exc:
            st.error(f"加载失败：{exc}")

df: pd.DataFrame | None = st.session_state.df

# ============ ② 数据画像 ============
if df is not None:
    st.header("② 数据画像")
    st.markdown(f"文件 `{st.session_state.df_name}` · SHA256 `{st.session_state.df_sha[:16]}…`")
    prof = profile_table(df)
    m1, m2, m3 = st.columns(3)
    m1.metric("测量行数 n_rows", prof["n_rows"])
    m2.metric("列数", prof["n_columns"])
    unit_col = st.session_state.get("unit_col")
    m3.metric(
        "独立实验单位数 n_units",
        count_units(df, unit_col) if unit_col in set(df.columns) else "见③设计确认",
        help="系统绝不默认『一行=一个独立实验单位』；先在③完成单位确认。",
    )
    st.dataframe(df.head(20), use_container_width=True)
else:
    st.info("载入演示数据或上传文件后，这里将展示数据画像。")
    st.stop()

# ============ ③ 设计确认 ============
st.header("③ 研究设计确认（推断检验的前置条件）")
st.warning(f"**系统提问**：{QUESTION_UNIT_MEANING}", icon="❓")
st.caption("独立性由研究设计决定，只能由您确认；系统不根据列名自动推断。未完成确认前，组间推断性检验一律阻断。")

c1, c2 = st.columns(2)
with c1:
    group_col = st.selectbox("分组列（group）", df.columns, index=0)
    value_options = [
        c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])
    ] or list(df.columns)
    outcome_col = st.selectbox("数值结局列（value）", value_options, index=0)
with c2:
    unit_col = st.selectbox(
        "独立实验单位列（unit_id）",
        df.columns,
        help="处理独立施加的对象标识列（如 fermenter_id/batch_id）。",
    )
    tech_col = st.selectbox(
        "技术重复列（可空）", ["（无）"] + list(df.columns),
        help="同一单位的多次技术测量标识；没有请选（无）。",
    )

design_type = st.radio(
    "研究设计类型",
    [
        DesignType.INDEPENDENT_GROUPS.value,
        DesignType.PAIRED.value,
        DesignType.UNKNOWN.value,
    ],
    format_func={
        DesignType.INDEPENDENT_GROUPS.value: "独立两组（处理施加在不同单位上）",
        DesignType.PAIRED.value: "配对设计（需 pair_id 列）",
        DesignType.UNKNOWN.value: "我不确定（保持阻断）",
    }.get,
)
pair_col: str | None = None
if design_type == DesignType.PAIRED.value:
    pair_col = st.selectbox("配对标识列（pair_id）", df.columns, index=0)

unit_semantics = st.text_area(
    "实验单位代表什么（必填后才能确认）",
    placeholder="例如：每一行是一个独立发酵罐批次；同一批次内的 3 次测量是技术重复，不是独立样本。",
)
confirm_box = st.checkbox(
    "我确认上述列与单位语义如实反映研究设计（确认动作记入证据链）"
)

if st.button("保存设计确认", type="primary", disabled=not confirm_box):
    try:
        design = DesignContext(
            dataset_id=st.session_state.df_sha[:16],
            group_column=group_col,
            outcome_column=outcome_col,
            independent_unit_column=unit_col,
            technical_repeat_column=None if tech_col == "（无）" else tech_col,
            pair_id_column=pair_col,
            design_type=DesignType(design_type),
            independence_confirmed=bool(unit_semantics.strip()) and design_type != DesignType.UNKNOWN.value,
            confirmed_by_user=bool(unit_semantics.strip()),
            unit_semantics=unit_semantics.strip(),
            confirmation_timestamp=now_utc(),
        )
        design.validate_against(df)
        st.session_state.design = design
        st.session_state.unit_col = unit_col
        st.session_state.review = None  # 设计已变更，旧审查结果作废
        st.success("设计确认已保存（含时间戳），进入证据链。")
    except Exception as exc:
        st.error(f"设计确认失败：{exc}")

design: DesignContext | None = st.session_state.get("design")
if design is not None:
    st.session_state.unit_col = design.independent_unit_column
    n_rows = len(df)
    n_units = count_units(df, design.independent_unit_column)
    k1, k2, k3 = st.columns(3)
    k1.metric("测量行数 n_rows", n_rows)
    k2.metric("独立实验单位数 n_units", n_units)
    k3.metric(
        "行数/单位数",
        f"{n_rows / n_units:.2f}" if n_units else "-",
        help=">1 提示可能存在技术重复或重复测量，M2 规则引擎将审查。",
    )
    if n_rows != n_units:
        st.info(
            f"检测到每个单位平均 {n_rows / n_units:.2f} 行：多行共享同一单位。"
            "在完成技术重复确认前，行级推断检验保持阻断（M2 风险规则）。"
        )

# ============ ④ 设计风险审查 ============
st.header("④ 设计风险审查（六条规则）")

if design is not None:
    if st.button("运行设计风险审查", type="primary"):
        st.session_state.review = review_design(df, design)

    review: ReviewReport | None = st.session_state.get("review")
    if review is not None:
        if review.overall == Severity.BLOCK:
            st.error(f"审查结论：**BLOCK** —— 存在 {len(review.open_blockers)} 项未解除的阻断，"
                     "推断性检验被禁止。")
        elif review.overall == Severity.WARN:
            st.warning("审查结论：**WARN** —— 可继续分析，但存在需要知悉的风险。")
        else:
            st.success("审查结论：**PASS** —— 设计确认完整，可执行白名单统计。")

        badge = {Severity.BLOCK: "🔴", Severity.WARN: "🟡", Severity.PASS: "🟢"}
        for f in review.findings:
            resolved = f.resolution_status == Resolution.RESOLVED
            box = st.success if f.severity == Severity.PASS else (
                st.error if f.severity == Severity.BLOCK else st.warning
            )
            with box(f"{badge[f.severity]} {f.rule_id}"
                     + ("（已解除）" if resolved else "")):
                st.markdown(f.explanation)
                st.markdown(f"**需要做什么**：{f.required_action}")
                if f.evidence_columns:
                    st.caption(f"证据列：{', '.join(f.evidence_columns)}"
                               f" · 涉及单位 {f.affected_unit_count} 个"
                               + (f" · 涉及行 {f.n_rows_involved}" if f.n_rows_involved else ""))

        # —— TECH_REP_RISK 解除路径：确认技术重复并授权聚合 ——
        tech_blocked = any(
            f.rule_id == "TECH_REP_RISK" and f.resolution_status == Resolution.OPEN
            for f in review.findings
        )
        if tech_blocked:
            with st.expander("解除路径：确认技术重复并授权按单位均值聚合", expanded=True):
                tech_col_ui = st.selectbox(
                    "技术重复标识列（同一单位内的多次测量）",
                    ["（请选择）"] + list(df.columns),
                )
                st.caption("确认后系统将按 unit×group 取均值，有效样本量=独立单位数"
                           "（如 24 行 → 8 单位）；原始行数与单位数都保留在报告中。")
                if st.button("我确认多行为技术重复，授权按单位均值聚合", type="primary"):
                    if tech_col_ui == "（请选择）":
                        st.error("请先选择技术重复标识列。")
                    else:
                        from sciverify.design.schema import AggregationPolicy

                        st.session_state.design = design.model_copy(
                            update={
                                "technical_repeat_column": tech_col_ui,
                                "aggregation_policy": AggregationPolicy.MEAN_BY_UNIT,
                            }
                        )
                        st.session_state.review = review_design(
                            df, st.session_state.design
                        )
                        st.rerun()
else:
    st.caption("完成③设计确认后才能运行风险审查（EU_UNKNOWN 是第一条规则）。")

# ============ ④.5 智能体执行（Intern-S2 规划 / REPLAY） ============
st.header("④½ 智能体执行（规划 → 工具 → 核验）")
review_obj: ReviewReport | None = st.session_state.get("review")
if design is not None and review_obj is not None and review_obj.overall != Severity.BLOCK:
    wf_mode = "LIVE" if has_live_api() else "REPLAY_OFFLINE"
    if wf_mode == "LIVE":
        st.success("运行模式：**LIVE Intern-S2**（书生真实 API 做任务理解与工具选择）")
    else:
        st.warning(
            "运行模式：**REPLAY_OFFLINE**（无 INTERN_API_KEY，使用确定性规则规划器；"
            "不冒充 LIVE 模型调用）"
        )
    question_ui = st.text_input(
        "你的分析问题（自然语言）",
        value=f"比较两组{design.outcome_column}是否有差异",
    )
    if st.button("运行智能体工作流", type="primary"):
        from sciverify.agents.workflow import run_workflow
        from sciverify.api.intern_client import InternClient
        from sciverify.tools import stats as st_stats

        try:
            state = run_workflow(
                df,
                design,
                question=question_ui,
                mode=wf_mode,
                client=InternClient() if wf_mode == "LIVE" else None,
            )
            st.session_state.wf_state_events = state.events
            st.session_state.wf_status = state.status
            st.session_state.wf_reason = state.blocked_reason
            st.session_state.wf_plan = state.plan
            st.session_state.wf_plan_source = state.plan_source
            st.session_state.wf_verification = state.verification
            if state.tool_result and state.unit_df is not None:
                import uuid as _u2

                png2 = LOG_DIR / f"boxplot_{_u2.uuid4().hex[:8]}.png"
                st_stats.draw_boxplot(
                    state.unit_df, design.group_column, design.outcome_column,
                    str(png2),
                    subtitle=f"workflow n_units={state.tool_result['n_units_used']}",
                )
                st.session_state.boxplot_path = str(png2)
                st.session_state.tool_result = state.tool_result
        except Exception as exc:
            st.error(f"工作流失败：{exc}")
            st.session_state.wf_status = "FAILED"

    events_ui = st.session_state.get("wf_state_events")
    if events_ui:
        st.markdown(f"**终态：`{st.session_state.get('wf_status')}`**")
        if st.session_state.get("wf_reason"):
            st.error(f"阻断原因：{st.session_state['wf_reason'][:300]}")
        for ev in events_ui:
            icon = {"OK": "✅", "PASS": "✅", "BLOCK": "⛔", "BLOCKED": "⛔",
                    "WARN": "⚠️", "REJECTED": "⛔", "FAIL": "⛔",
                    "SKIPPED": "⏭️", "FALLBACK": "🔁", "EXHAUSTED": "⛔"}.get(ev["status"], "•")
            st.markdown(
                f"{icon} `{ev['node']}` — **{ev['status']}** "
                f"`{ {k: v for k, v in ev.items() if k not in ('ts', 'node', 'status')} }`"
            )
else:
    st.caption("完成③④并解除 BLOCK 后，可运行智能体工作流（无 Key 时为 REPLAY 离线模式）。")

# ============ ⑤ 统计执行（确定性白名单工具） ============
st.header("⑤ 统计结果（确定性计算）")

if design is not None and review_obj is not None and review_obj.overall != Severity.BLOCK:
    expected_tool = (
        "paired_t_test" if design.design_type == DesignType.PAIRED else "welch_t_test"
    )
    st.markdown(
        f"审查通过（{review_obj.overall.value}）。按设计类型将执行："
        f"**`{expected_tool}`**（有效样本量 = 独立实验单位数）。"
    )
    if st.button("执行统计（本地确定性计算）", type="primary"):
        from sciverify.tools import stats as st_stats
        from sciverify.tools.registry import default_registry

        try:
            unit_df = st_stats.prepare_analysis_frame(df, design)
            reg = default_registry()
            if expected_tool == "welch_t_test":
                tool_result = reg.execute(
                    "welch_t_test",
                    {
                        "group_col": design.group_column,
                        "value_col": design.outcome_column,
                    },
                    unit_df,
                    design=design,
                )
            else:
                tool_result = reg.execute(
                    "paired_t_test",
                    {
                        "group_col": design.group_column,
                        "value_col": design.outcome_column,
                        "pair_col": design.pair_id_column,
                    },
                    unit_df,
                    design=design,
                )
            import uuid as _uuid

            png_path = LOG_DIR / f"boxplot_{_uuid.uuid4().hex[:8]}.png"
            st_stats.draw_boxplot(
                unit_df, design.group_column, design.outcome_column,
                str(png_path),
                subtitle=f"n_rows={len(df)} → n_units={tool_result['n_units_used']}",
            )
            st.session_state.tool_result = tool_result
            st.session_state.boxplot_path = str(png_path)
        except Exception as exc:
            st.error(f"统计执行失败：{exc}")

    tool_result = st.session_state.get("tool_result")
    if tool_result:
        cA, cB = st.columns([3, 2])
        with cA:
            st.subheader(f"{tool_result['tool']} 结果")
            m1, m2, m3 = st.columns(3)
            m1.metric("有效样本量 n_units", tool_result["n_units_used"])
            m2.metric("统计量", f"{tool_result['statistic']:.4f}")
            m3.metric("P 值", f"{tool_result['p_value']:.4g}")
            st.markdown(
                f"- 各组单位数：{tool_result['n_by_group']}\n"
                f"- 各组均值：{ {k: round(v, 4) for k, v in tool_result['mean_by_group'].items()} }\n"
                f"- 均值差：{tool_result['mean_difference']:.4f}"
                f"（95% CI [{tool_result['confidence_interval'][0]:.4f}, "
                f"{tool_result['confidence_interval'][1]:.4f}]）\n"
                f"- 差值定义：{tool_result['difference_definition']}"
            )
            for w in tool_result["warnings"]:
                st.warning(w)
            st.caption(
                "数值全部来自本地 SciPy/NumPy 确定性计算："
                f"{tool_result['library_versions']}"
            )
        with cB:
            st.image(st.session_state.get("boxplot_path"))
elif design is not None:
    st.info("④ 审查存在未解除的 BLOCK 或尚未运行——统计执行被阻断。")
else:
    st.caption("完成③④后在此执行统计（M4 将接入 Intern-S2 工具规划）。")

# ============ ⑥ 证据导出（报告 + 复现 ZIP） ============
st.header("⑥ 证据导出")

if design is not None and st.session_state.get("tool_result"):
    if st.button("生成 HTML 报告与复现 ZIP", type="primary"):
        from sciverify.reports.html_report import build_report_for_state, save_report
        from sciverify.reports.repro_zip import build_repro_zip
        from sciverify.storage.run_store import sha256_of
        from sciverify.verification.claim_guard import check_report_against_result

        tr = st.session_state.tool_result
        art_dir = LOG_DIR / "artifacts"
        art_dir.mkdir(parents=True, exist_ok=True)
        # 直接用当前 ToolResult 渲染（工作流 state 可能不存在——⑤ 手动路径）
        from sciverify.reports.html_report import render_report_html
        import datetime as _dt

        html = render_report_html(
            run_id=st.session_state.get("wf_run_id", "ui_manual"),
            mode="LIVE" if has_live_api() else "REPLAY_OFFLINE",
            dataset_name=st.session_state.df_name,
            dataset_sha256=st.session_state.df_sha,
            n_rows=int(len(df)),
            n_units=int(df[design.independent_unit_column].nunique()),
            design=design.snapshot(),
            review_overall=(review_obj.overall.value if review_obj else "UNKNOWN"),
            n_units_by_group={
                str(g): int(n)
                for g, n in df.groupby(design.group_column)[design.independent_unit_column].nunique().items()
            },
            findings=[f.model_dump(mode="json") for f in (review_obj.findings if review_obj else [])],
            plan=st.session_state.get("wf_plan"),
            plan_source=st.session_state.get("wf_plan_source", "ui_manual"),
            tool_result=tr,
            verification=st.session_state.get("wf_verification") or {"status": "MANUAL", "checks": {}},
            boxplot_png_path=st.session_state.get("boxplot_path"),
            result_hash=sha256_of(tr),
        )
        guard = check_report_against_result(html, tr)
        if guard["status"] != "PASS":
            st.error(f"报告数值绑定失败，拒绝导出：{guard['mismatches']}")
        else:
            ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = save_report(html, art_dir / f"report_{ts}.html")
            zip_path = build_repro_zip(
                art_dir / f"repro_{ts}.zip",
                run_id=f"ui_{ts}",
                data_path=DEMOS_DIR / st.session_state.df_name
                if (DEMOS_DIR / st.session_state.df_name).exists()
                else st.session_state.get("upload_source", ""),
                design_snapshot=design.snapshot(),
                manifest={"run_id": f"ui_{ts}", "mode": "UI_MANUAL",
                          "result_hash": sha256_of(tr)},
                events=[{"node": "UI", "status": "MANUAL_RUN"}],
                tool_result=tr,
                report_html=html,
                created_at=ts,
            )
            st.session_state.report_path = str(report_path)
            st.session_state.zip_path = str(zip_path)
            st.success(f"报告与复现包已生成（claim_guard=PASS）：{report_path.name}")

    if st.session_state.get("report_path"):
        st.download_button(
            "⬇️ 下载 HTML 报告",
            data=Path(st.session_state.report_path).read_bytes(),
            file_name=Path(st.session_state.report_path).name,
            mime="text/html",
        )
        if st.session_state.get("zip_path") and Path(st.session_state.zip_path).exists():
            st.download_button(
                "⬇️ 下载复现 ZIP（数据+设计+结果+报告）",
                data=Path(st.session_state.zip_path).read_bytes(),
                file_name=Path(st.session_state.zip_path).name,
                mime="application/zip",
            )
elif design is None:
    st.caption("完成③④⑤后可导出证据（报告数值由 claim_guard 与 ToolResult 强一致校验）。")
