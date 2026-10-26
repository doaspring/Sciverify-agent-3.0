"""独立数值核验（V3 方案 §3.5 核心能力 C）.

与 tools/stats.py 完全独立的第二实现：
- 不 import scipy/numpy 的统计函数，只用 math 标准库；
- t 分布 P 值用正则化不完全 Beta 函数（ modified Lentz 连分式，
  Numerical Recipes 经典算法）自行实现；
- 对 ToolResult 逐字段复算并给出 PASS/FAIL（超容差即 FAIL → 阻断报告）。
"""
from __future__ import annotations

import math
from typing import Any

TOL_STAT = 1e-8       # 统计量/均值差相对容差
TOL_P = 1e-6          # P 值绝对容差（不同算法路径的数值稳定性）


# ------------------------------------------------ 不完全 Beta（独立实现）


def _betacf(a: float, b: float, x: float) -> float:
    """正则化不完全 Beta 的连分式（modified Lentz，迭代上限 200）。"""
    eps, fpmin = 3.0e-12, 1.0e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for m in range(1, 201):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def regularized_beta(a: float, b: float, x: float) -> float:
    """I_x(a, b) 正则化不完全 Beta 函数。"""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    ln_bt = (
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    bt = math.exp(ln_bt)
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def t_two_sided_p(t: float, df: float) -> float:
    """Student t 双侧 P 值（独立于 SciPy）。"""
    if df <= 0:
        return float("nan")
    x = df / (df + t * t)
    return regularized_beta(df / 2.0, 0.5, x)


def t_ppf_two_sided(alpha: float, df: float) -> float:
    """双侧临界值 t_{1-α/2, df}：对 t_two_sided_p 做二分反解。"""
    target = alpha
    lo, hi = 0.0, 1e4
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if t_two_sided_p(mid, df) > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ------------------------------------------------ Welch 复算


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def _var(xs: list[float]) -> float:
    m = _mean(xs)
    return sum((v - m) ** 2 for v in xs) / (len(xs) - 1)


def verify_welch(
    samples_by_group: dict[str, list[float]], result: dict[str, Any]
) -> dict[str, Any]:
    """用独立公式复算 ToolResult（welch_t_test）并逐字段比对。"""
    g1, g2 = sorted(samples_by_group)
    a, b = samples_by_group[g1], samples_by_group[g2]
    m1, m2 = _mean(a), _mean(b)
    v1, v2 = _var(a), _var(b)
    n1, n2 = len(a), len(b)
    se = math.sqrt(v1 / n1 + v2 / n2)
    t_stat = (m2 - m1) / se
    df_w = se**4 / (v1**2 / (n1**2 * (n1 - 1)) + v2**2 / (n2**2 * (n2 - 1)))
    p = t_two_sided_p(t_stat, df_w)
    tcrit = t_ppf_two_sided(0.05, df_w)
    diff = m2 - m1

    checks = {
        "statistic": abs(result["statistic"] - t_stat) <= TOL_STAT * max(1.0, abs(t_stat)),
        "p_value": abs(result["p_value"] - p) <= TOL_P,
        "mean_difference": abs(result["mean_difference"] - diff)
        <= TOL_STAT * max(1.0, abs(diff)),
        "ci_low": abs(result["confidence_interval"][0] - (diff - tcrit * se))
        <= 1e-4 * max(1.0, abs(diff)),
        "ci_high": abs(result["confidence_interval"][1] - (diff + tcrit * se))
        <= 1e-4 * max(1.0, abs(diff)),
        "n_units": result["n_units_used"] == n1 + n2,
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "independent": {
            "t": t_stat,
            "df": df_w,
            "p": p,
            "ci": (diff - tcrit * se, diff + tcrit * se),
            "n1": n1,
            "n2": n2,
        },
    }


# ------------------------------------------------ 配对复算


def verify_paired(
    diffs_sorted_by_pair: list[float], result: dict[str, Any]
) -> dict[str, Any]:
    """d 列表 = 按配对顺序的（后组-前组）差值；独立复算配对 t 全字段。"""
    d = diffs_sorted_by_pair
    n = len(d)
    md = _mean(d)
    sd = math.sqrt(_var(d))
    se = sd / math.sqrt(n)
    t_stat = md / se
    p = t_two_sided_p(t_stat, n - 1)
    tcrit = t_ppf_two_sided(0.05, n - 1)

    checks = {
        "statistic": abs(result["statistic"] - t_stat) <= TOL_STAT * max(1.0, abs(t_stat)),
        "p_value": abs(result["p_value"] - p) <= TOL_P,
        "mean_difference": abs(result["mean_difference"] - md)
        <= TOL_STAT * max(1.0, abs(md)),
        "ci_low": abs(result["confidence_interval"][0] - (md - tcrit * se))
        <= 1e-4 * max(1.0, abs(md)),
        "ci_high": abs(result["confidence_interval"][1] - (md + tcrit * se))
        <= 1e-4 * max(1.0, abs(md)),
        "n_pairs": result["n_units_used"] == n,
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "independent": {"t": t_stat, "p": p, "mean_d": md, "ci": (md - tcrit * se, md + tcrit * se), "n": n},
    }


# ------------------------------------------------ 描述统计复算


def verify_describe(
    values_by_group: dict[str, list[float]], result: dict[str, Any]
) -> dict[str, Any]:
    ok = True
    for g, vals in values_by_group.items():
        m = _mean(vals)
        if abs(result[g]["mean"] - m) > TOL_STAT * max(1.0, abs(m)):
            ok = False
        if result[g]["n"] != len(vals):
            ok = False
    return {"status": "PASS" if ok else "FAIL", "checks": {"means_and_n": ok}}
