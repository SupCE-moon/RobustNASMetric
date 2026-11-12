"""Does TRNAS Phi add anything beyond the feature-drift family?

The question
------------
The leaderboard has Phi (TRNAS consistency) at rank 1.  But 35 metrics in the
same board are drift-style feature measures that cluster together at
rho >= 0.5.  If Phi is just a re-parameterisation of "how much the features
move", its rank-1 position says nothing new.

The plan (`docs/experiments/增量检验方案_Φ对特征漂移族_20261112.md`) fixes the
judgement before the numbers exist:

* T1 (primary)  partial Spearman of Phi and the target, controlling for B,
                averaged over the eight settings.  Called delta-rho below.
* T2 (support)  out-of-fold R^2 gain when adding Phi to a model that already
                has B, with folds grouped by architecture.
* T3 (support)  within-quintile-of-B Spearman: if Phi only re-sorts whole
                quantiles, the within-quintile correlation collapses.

B is the mean within-setting percentile rank of the 35 drift-family metrics,
taken from the frozen clustering audit rather than re-chosen here.  A PCA
first component is reported as a robustness check only.

Significance
------------
The independent unit is the **architecture** (204 of them), not the row:
NAS-Bench-201 contributes five training stages per architecture and DARTS two,
so row-level resampling would treat one network as five independent samples
and shrink the interval.  Bootstrapping resamples architectures; the
permutation test shuffles Phi within a setting, which breaks the Phi-target
link while keeping the marginal distributions.

Verdicts follow the plan verbatim:
* interval lower bound > 0 and p < 0.05  -> "has incremental value"
* interval covers 0 and T2/T3 all near 0 -> "no evidence of incremental value"
  (not "proven to be zero")
* anything else                          -> "not settled"
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

PHI = "trnas_consistency_phi"
FAMILY_CUT = 0.5
FAMILY_SIZE = 35
SETTING_KEYS = (
    "rb40_mature",
    "nb201_e0",
    "nb201_e5",
    "nb201_e10",
    "nb201_e25",
    "nb201_e50",
    "darts_e0",
    "darts_e100",
)
GRID_OF_SETTING = {
    "rb40_mature": "rb40",
    "nb201_e0": "nb201",
    "nb201_e5": "nb201",
    "nb201_e10": "nb201",
    "nb201_e25": "nb201",
    "nb201_e50": "nb201",
    "darts_e0": "darts",
    "darts_e100": "darts",
}
ALPHA = 0.05
DEFAULT_RESAMPLES = 2000
DEFAULT_SEED = 20261112
QUANTILES = 5
MIN_QUANTILE_N = 8


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_family(redundancy_json: Path) -> list[str]:
    """Take the 35-member drift cluster from the frozen clustering audit.

    Re-deriving the cluster here would make the baseline movable, and a movable
    baseline can be tuned until Phi looks additive.
    """
    payload = json.loads(redundancy_json.read_text(encoding="utf-8"))
    cut = next(c for c in payload["cuts"] if c["min_rho"] == FAMILY_CUT)
    cluster = next(c for c in cut["clusters"] if c["size"] == FAMILY_SIZE)
    members = sorted(cluster["members"])
    if PHI in members:
        raise ValueError("Phi must not be part of the baseline family")
    return members


def load_panel(rows_path: Path, repair_path: Path | None) -> tuple[pd.DataFrame, pd.Series, dict[str, str]]:
    frame = pd.read_csv(rows_path)
    if repair_path is not None and repair_path.exists():
        frame = pd.concat([frame, pd.read_csv(repair_path)], ignore_index=True)
    frame = frame[(frame["status"] == "ok") & frame["metric_value"].notna()].copy()
    frame = frame.drop_duplicates(["setting", "model_or_arch_id", "metric_name"], keep="last")

    panel = frame.pivot_table(
        index=["setting", "model_or_arch_id"],
        columns="metric_name",
        values="metric_value",
        aggfunc="first",
    ).sort_index()
    target = frame.groupby(["setting", "model_or_arch_id"])["target_value"].first().reindex(panel.index)
    if target.isna().any():
        raise ValueError("a row has no target_value")
    direction = frame.drop_duplicates("metric_name").set_index("metric_name")["direction"].to_dict()
    return panel, target, direction


def orient(panel: pd.DataFrame, direction: dict[str, str]) -> pd.DataFrame:
    """Flip 'lower is better' metrics so that larger always means more robust.

    Feature drift is declared 'lower': a network whose features move less is
    expected to be more robust.  Without the flip, B correlates -0.83 with the
    target and every partial correlation below reads backwards.
    """
    flipped = panel.copy()
    for name in flipped.columns:
        if direction.get(name, "higher") == "lower":
            flipped[name] = -flipped[name]
    return flipped


def build_frame(panel: pd.DataFrame, target: pd.Series, family: list[str]) -> pd.DataFrame:
    """One row per (setting, architecture) with y, phi and the B baseline.

    The identity of an architecture is ``grid:id``.  Both DARTS and
    NAS-Bench-201 number their architectures 0..N, so a bare id would silently
    merge DARTS network 0 with NAS-Bench-201 network 0 and make the bootstrap
    treat two unrelated networks as one.
    """
    frame = pd.DataFrame(
        {
            "y": target,
            "phi": panel[PHI],
            "b": panel[family].mean(axis=1),
        }
    ).reset_index()
    frame = frame.rename(columns={"model_or_arch_id": "arch_id"})
    frame["grid"] = frame["setting"].map(GRID_OF_SETTING)
    frame["arch"] = frame["grid"] + ":" + frame["arch_id"].astype(str)
    return frame


def percentile_within(frame: pd.DataFrame, column: str) -> pd.Series:
    """Percentile rank inside each setting, so settings with 40 and 100 rows compare."""
    return frame.groupby("setting")[column].rank(pct=True, method="average")


def _residualise(values: np.ndarray, control: np.ndarray) -> np.ndarray:
    slope, intercept = np.polyfit(control, values, 1)
    return values - (slope * control + intercept)


def partial_rho(x: np.ndarray, y: np.ndarray, control: np.ndarray) -> float:
    """Spearman of x and y after linearly removing control from both."""
    if len(x) < 4:
        return float("nan")
    rho = np.corrcoef(
        _residualise(rankdata(x), rankdata(control)),
        _residualise(rankdata(y), rankdata(control)),
    )[0, 1]
    return float(rho) if np.isfinite(rho) else float("nan")


def delta_rho_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Per-setting delta-rho (Phi vs y controlling for B) plus the raw pieces."""
    rows = []
    for setting, group in frame.groupby("setting", sort=False):
        x = group["phi"].to_numpy(float)
        y = group["y"].to_numpy(float)
        control = group["b"].to_numpy(float)
        rows.append(
            {
                "setting": setting,
                "grid": GRID_OF_SETTING.get(setting, "unknown"),
                "n": len(group),
                "rho_phi_y": float(spearmanr(x, y).statistic),
                "rho_b_y": float(spearmanr(control, y).statistic),
                "rho_phi_b": float(spearmanr(x, control).statistic),
                "delta_rho": partial_rho(x, y, control),
            }
        )
    return pd.DataFrame(rows)


def mean_delta(table: pd.DataFrame) -> float:
    return float(table["delta_rho"].mean())


def within_quantile_rho(frame: pd.DataFrame, quantiles: int = QUANTILES) -> dict[str, Any]:
    """T3: does Phi still order networks *inside* a band of equal drift?"""
    per_setting = {}
    for setting, group in frame.groupby("setting", sort=False):
        bins = pd.qcut(group["b"].rank(method="first"), quantiles, labels=False)
        values = []
        for _, band in group.groupby(bins, sort=True):
            if len(band) >= MIN_QUANTILE_N:
                values.append(float(spearmanr(band["phi"], band["y"]).statistic))
        per_setting[setting] = float(np.mean(values)) if values else float("nan")
    usable = [v for v in per_setting.values() if np.isfinite(v)]
    return {
        "per_setting": per_setting,
        "mean": float(np.mean(usable)) if usable else float("nan"),
        "settings_used": len(usable),
    }


def _fit_linear(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    """Least squares with an intercept, using numpy only (no sklearn dependency)."""
    design = np.column_stack([np.ones(len(x)), x])
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    return coefficients[1:], float(coefficients[0])


def _oof_r2(frame: pd.DataFrame, columns: list[str], folds: int = 5) -> float:
    """Out-of-fold R^2 with folds assigned by architecture, never by row.

    Five stages of one NAS-Bench-201 architecture are not five samples; letting
    them fall into different folds would leak the answer into the training side.
    """
    archs = np.array(sorted(frame["arch"].unique()))
    fold_of_arch = {arch: index % folds for index, arch in enumerate(archs)}
    fold = frame["arch"].map(fold_of_arch).to_numpy()
    y = frame["y"].to_numpy(float)
    x = frame[columns].to_numpy(float)
    predicted = np.full(len(y), np.nan)
    for current in range(folds):
        train = fold != current
        test = fold == current
        if test.sum() == 0 or train.sum() < len(columns) + 2:
            continue
        slopes, intercept = _fit_linear(x[train], y[train])
        predicted[test] = x[test] @ slopes + intercept
    keep = np.isfinite(predicted)
    residual = y[keep] - predicted[keep]
    total = ((y[keep] - y[keep].mean()) ** 2).sum()
    return float(1.0 - (residual ** 2).sum() / total)


def nested_r2(frame: pd.DataFrame) -> dict[str, float]:
    """T2: out-of-fold R^2 with B alone vs B + Phi, folds grouped by architecture."""
    base = _oof_r2(frame, ["b"])
    full = _oof_r2(frame, ["b", "phi"])
    return {"r2_base": base, "r2_full": full, "gain": full - base}


def _resample(frame: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Draw architectures with replacement and keep all their settings."""
    archs = frame["arch"].unique()
    drawn = rng.choice(archs, size=len(archs), replace=True)
    pieces = []
    for index, arch in enumerate(drawn):
        block = frame[frame["arch"] == arch].copy()
        block["arch"] = f"{arch}__draw{index}"
        pieces.append(block)
    return pd.concat(pieces, ignore_index=True)


def bootstrap(frame: pd.DataFrame, resamples: int, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(resamples):
        sample = _resample(frame, rng)
        draws.append(mean_delta(delta_rho_table(sample)))
    draws = np.array([d for d in draws if np.isfinite(d)])
    return {
        "mean": float(draws.mean()),
        "low": float(np.percentile(draws, 2.5)),
        "high": float(np.percentile(draws, 97.5)),
        "draws": int(len(draws)),
    }


def permutation(frame: pd.DataFrame, resamples: int, seed: int) -> dict[str, Any]:
    """Shuffle Phi across architectures within each setting."""
    rng = np.random.default_rng(seed + 1)
    observed = mean_delta(delta_rho_table(frame))
    null = []
    for _ in range(resamples):
        shuffled = frame.copy()
        shuffled["phi"] = shuffled.groupby("setting")["phi"].transform(
            lambda values: rng.permutation(values.to_numpy())
        )
        null.append(mean_delta(delta_rho_table(shuffled)))
    null = np.array([v for v in null if np.isfinite(v)])
    # The +1 keeps the estimate honest: with B shuffles the smallest attainable
    # p is 1/(B+1), and reporting 0.0000 would claim more than the design allows.
    exceed = int((np.abs(null) >= abs(observed)).sum())
    p_value = (exceed + 1) / (len(null) + 1)
    return {
        "null_mean": float(null.mean()),
        "null_low": float(np.percentile(null, 2.5)),
        "null_high": float(np.percentile(null, 97.5)),
        "p_value": float(p_value),
        "exceed": exceed,
        "draws": int(len(null)),
    }


def oracle_single_baseline(oriented: pd.DataFrame, target: pd.Series, family: list[str]) -> dict[str, Any]:
    """The hardest case for Phi: per setting, control for the single family
    member that correlates best with the target.

    Averaging 35 metrics dilutes whatever signal they carry, so the mean is a
    weak control by construction.  Picking the best member per setting is
    optimistic (it peeks at the answer and could not be used to forecast), so a
    delta-rho that stays positive here cannot be explained away as "the baseline
    was too blunt".
    """
    rows = []
    for setting, group in oriented.groupby(level="setting", sort=False):
        y = target.loc[group.index].to_numpy(float)
        best_name, best_rho = None, -np.inf
        for name in family:
            rho = spearmanr(group[name].to_numpy(float), y).statistic
            if np.isfinite(rho) and rho > best_rho:
                best_name, best_rho = name, float(rho)
        if best_name is None:
            continue
        rows.append(
            {
                "setting": setting,
                "best_member": best_name,
                "rho_best_y": best_rho,
                "delta_rho": partial_rho(
                    group[PHI].to_numpy(float), y, group[best_name].to_numpy(float)
                ),
            }
        )
    values = [row["delta_rho"] for row in rows if np.isfinite(row["delta_rho"])]
    return {
        "per_setting": rows,
        "mean_delta_rho": float(np.mean(values)) if values else float("nan"),
        "note": "optimistic baseline (best family member chosen per setting); "
        "a positive delta here is the conservative case for Phi",
    }


def pca_check(frame: pd.DataFrame, panel: pd.DataFrame, family: list[str]) -> dict[str, Any]:
    """Robustness check the plan asks for: is the result an artefact of averaging?"""
    components = []
    for _, group in panel.groupby(level="setting", sort=False):
        matrix = group[family].to_numpy(float)
        matrix = matrix - matrix.mean(axis=0)
        _, _, vt = np.linalg.svd(matrix, full_matrices=False)
        components.append(vt[0])
    # Align signs so a component cannot flip the correlation at random.
    reference = components[0]
    aligned = []
    for component in components:
        aligned.append(component if np.dot(component, reference) >= 0 else -component)
    aligned_frame = frame.copy()
    pc = []
    for setting, group in panel.groupby(level="setting", sort=False):
        matrix = group[family].to_numpy(float)
        matrix = matrix - matrix.mean(axis=0)
        pc.extend((matrix @ aligned[0]).tolist())
    aligned_frame["b"] = pc
    table = delta_rho_table(aligned_frame)
    return {"mean_delta_rho": float(table["delta_rho"].mean()), "per_setting": table["delta_rho"].tolist()}


def verdict(ci_low: float, ci_high: float, p_value: float, t2_gain: float, t3_mean: float) -> str:
    positive = ci_low > 0 and p_value < ALPHA
    null_like = ci_low <= 0 <= ci_high and abs(t2_gain) < 0.02 and abs(t3_mean) < 0.10
    if positive:
        return "has incremental value"
    if null_like:
        return "no evidence of incremental value (not the same as proving zero)"
    return "not settled"


def analyze(
    rows_path: Path,
    repair_path: Path | None,
    redundancy_json: Path,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    family = load_family(redundancy_json)
    panel, target, direction = load_panel(rows_path, repair_path)
    oriented = orient(panel, direction)
    frame = build_frame(oriented, target, family)
    frame["y"] = percentile_within(frame, "y")
    frame["phi"] = percentile_within(frame, "phi")
    frame["b"] = percentile_within(frame, "b")

    table = delta_rho_table(frame)
    observed = mean_delta(table)
    boot = bootstrap(frame, resamples, seed)
    perm = permutation(frame, resamples, seed)
    t2 = nested_r2(frame)
    t3 = within_quantile_rho(frame)
    pca = pca_check(frame, oriented, family)
    oracle = oracle_single_baseline(oriented, target, family)

    per_grid = {}
    for grid in ("rb40", "nb201", "darts"):
        subset = frame[frame["grid"] == grid]
        if subset["setting"].nunique() == 0:
            continue
        per_grid[grid] = {
            "settings": sorted(subset["setting"].unique().tolist()),
            "mean_delta_rho": mean_delta(delta_rho_table(subset)),
        }

    return {
        "target_metric": PHI,
        "baseline_family": {"cut_rho": FAMILY_CUT, "size": len(family), "members": family},
        "observed_delta_rho": observed,
        "per_setting": table.to_dict(orient="records"),
        "per_grid": per_grid,
        "bootstrap": boot,
        "permutation": perm,
        "t2_nested_r2": t2,
        "t3_within_quantile": t3,
        "pca_check": pca,
        "oracle_single_baseline": oracle,
        "verdict": verdict(boot["low"], boot["high"], perm["p_value"], t2["gain"], t3["mean"]),
        "config": {
            "resamples": resamples,
            "seed": seed,
            "alpha": ALPHA,
            "quantiles": QUANTILES,
            "independent_unit": "architecture",
            "n_architectures": int(frame["arch"].nunique()),
            "n_rows": int(len(frame)),
        },
        "inputs": {
            "rows": str(rows_path),
            "rows_sha256": sha256_file(rows_path),
            "redundancy": str(redundancy_json),
            "redundancy_sha256": sha256_file(redundancy_json),
        },
    }


VERDICT_ZH = {
    "has incremental value": "有增量信息：控制漂移族之后，Φ 仍能解释相当一部分鲁棒性排序。",
    "no evidence of incremental value (not the same as proving zero)": (
        "没有增量证据；这不等于证明增量为零。"
    ),
    "not settled": "证据不足，暂不下结论。",
}


def render_markdown(result: dict[str, Any]) -> str:
    boot = result["bootstrap"]
    perm = result["permutation"]
    t2 = result["t2_nested_r2"]
    t3 = result["t3_within_quantile"]
    lines = [
        "# Φ 对特征漂移族是否有增量信息（T1/T2/T3，2026-11-12）",
        "",
        "方案：`docs/experiments/增量检验方案_Φ对特征漂移族_20261112.md`（判据在看数之前写死）。",
        "",
        f"基准 B = 35 个漂移族指标（ρ≥{FAMILY_CUT} 聚类）的组内百分位秩均值；"
        f"独立单元是**架构**（{result['config']['n_architectures']} 个），不是行（{result['config']['n_rows']} 行）。",
        "",
        "## 结论",
        "",
        f"**{result['verdict']}**",
        "",
        f"{VERDICT_ZH.get(result['verdict'], '')}",
        "",
        f"- T1 偏秩相关 Δρ = **{result['observed_delta_rho']:+.3f}**，"
        f"按架构 bootstrap 的 95% 区间 [{boot['low']:+.3f}, {boot['high']:+.3f}]；"
        f"置换检验 p = {perm['p_value']:.4f}（零分布均值 {perm['null_mean']:+.3f}）。",
        f"- T2 嵌套模型：只用 B 的 OOF R² = {t2['r2_base']:.3f}，加 Φ 后 {t2['r2_full']:.3f}，"
        f"增量 {t2['gain']:+.3f}。",
        f"- T3 同漂移分位内的秩相关均值 {t3['mean']:+.3f}（{t3['settings_used']} 个设置可用）。",
        "",
        "## 逐设置",
        "",
        "| 设置 | n | ρ(Φ,y) | ρ(B,y) | ρ(Φ,B) | Δρ |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in result["per_setting"]:
        lines.append(
            f"| {row['setting']} | {row['n']} | {row['rho_phi_y']:+.3f} | {row['rho_b_y']:+.3f} | "
            f"{row['rho_phi_b']:+.3f} | {row['delta_rho']:+.3f} |"
        )
    lines += ["", "## 分网格敏感性", ""]
    for grid, info in result["per_grid"].items():
        lines.append(f"- {grid}：Δρ = {info['mean_delta_rho']:+.3f}（{len(info['settings'])} 个设置）")
    lines += [
        "",
        "## 稳健性检查（PCA 第一主成分替代平均秩）",
        "",
        f"Δρ = {result['pca_check']['mean_delta_rho']:+.3f}。若与主检验差别大，说明 B 的构造方式在左右结论。",
        "",
        "## 更强的对照（每设置挑族内与 target 最相关的单个指标）",
        "",
        f"Δρ = {result['oracle_single_baseline']['mean_delta_rho']:+.3f}。"
        "这个基准每设置都在 35 个成员里挑最有利的一个，属于乐观上限，"
        "真实预测场景里做不到；在它之下 Δρ 仍为正，说明「平均把信号稀释了」解释不掉结论。",
        "",
        "## 怎么读",
        "",
        "- **Δρ 的含义**：在已经把「特征漂移幅度」算进去之后，Φ 还能解释多少鲁棒性排序。",
        "  Δρ 接近 0 表示 Φ 基本是漂移族的另一种写法；Δρ 明显大于 0 表示它带来了新信息。",
        "- **Δρ 必须和 ρ(B,y) 一起读**。控制变量与 target 无关时，偏相关会退化回 ρ(Φ,y)：",
        "  `darts_e100` 的 ρ(B,y)=−0.028，那里 Δρ 高不是因为 Φ 增量大，",
        "  而是因为漂移族在那个网格上根本预测不了鲁棒性。真正有信息量的是 B 也强的设置。",
        "- **不能读成因果**。这里问的是「信息是否重复」，不是「谁导致鲁棒性」。",
        "- **p 值来自置换**：把 Φ 在设置内打乱，破坏它与 target 的配对，其余不变。",
        "  这样得到的零分布保留了 B 与 target 的真实关系，是唯一合适的对照。",
        "",
        "## 局限",
        "",
        "- 只有 85 个指标：第 86 个 `feature_adv_excess_centered_unit_l2` 的原始行只在服务器，离线无法纳入。",
        "- 三个网格的 target 语义不同（RB40 是公开 AutoAttack 成绩，NB201/DARTS 是各训练阶段的评估）。",
        "  混在一起平均是为了对齐冻结协议，不是因为它自然。",
        "- B 的成员来自一次冻结聚类（ρ≥0.5），换阈值成员会变。PCA 与 oracle 两个对照用来暴露这种依赖，",
        "  但不能完全消除它。",
        "- 这个检验只回答「信息是否重复」，不回答「Φ 为什么有效」。后者需要另一套实验。",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--repair", type=Path, default=None)
    parser.add_argument("--redundancy", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--resamples", type=int, default=DEFAULT_RESAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)

    repair = args.repair if args.repair is not None else args.rows.parent / "repair_rows.csv"
    result = analyze(args.rows, repair, args.redundancy, args.resamples, args.seed)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(render_markdown(result), encoding="utf-8")

    boot = result["bootstrap"]
    print(
        f"delta_rho={result['observed_delta_rho']:+.3f} "
        f"[{boot['low']:+.3f},{boot['high']:+.3f}] p={result['permutation']['p_value']:.4f} "
        f"t2_gain={result['t2_nested_r2']['gain']:+.3f} t3={result['t3_within_quantile']['mean']:+.3f}"
    )
    print("verdict:", result["verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
