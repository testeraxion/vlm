from __future__ import annotations

import numpy as np

VLM_SIGNALS = {
    "hazard_count": lambda r: float(len(r["vlm"].get("hazards", []))),
    "hallucination_risk": lambda r: float(r["vlm"].get("hallucination_risk", 0.0)),
    "consistency_score": lambda r: float(r["vlm"].get("consistency_score", 1.0)),
    "evidence_alignment": lambda r: float(r["explanation_evidence"].get("evidence_alignment_score", 0.0)),
}

SAFETY_METRICS = {
    "ate_rmse": lambda r: float(r["slam_metrics"]["ate_rmse"]),
    "drift_final_m": lambda r: float(r["slam_metrics"]["drift_final_m"]),
    "rpe_mean": lambda r: float(r["slam_metrics"]["rpe_mean"]),
    "violation_count": lambda r: float(r["runtime_verification"]["violation_count"]),
    "stl_robustness": lambda r: float(r["runtime_verification"]["stl_robustness"]),
}


def _clean(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    x, y = _clean(x, y)
    if len(x) < 2 or np.all(x == x[0]) or np.all(y == y[0]):
        return None
    xm, ym = x - x.mean(), y - y.mean()
    denom = np.sqrt((xm**2).sum() * (ym**2).sum())
    if denom == 0.0:
        return None
    return float(np.clip((xm * ym).sum() / denom, -1.0, 1.0))


def spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    x, y = _clean(x, y)
    if len(x) < 2:
        return None
    xr = np.argsort(np.argsort(x)).astype(float)
    yr = np.argsort(np.argsort(y)).astype(float)
    return pearson(xr, yr)


def _permutation_p_value(
    x: np.ndarray,
    y: np.ndarray,
    r_obs: float,
    n_perm: int = 2000,
    seed: int = 7,
) -> float:
    """Two-sided permutation p-value for the observed correlation coefficient."""
    x, y = _clean(x, y)
    if len(x) < 3:
        return None
    rng = np.random.default_rng(seed)
    abs_obs = abs(r_obs)
    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(y)
        r = pearson(x, perm)
        if r is not None and abs(r) >= abs_obs:
            count += 1
    return float((count + 1) / (n_perm + 1))


def correlate_vlm_with_safety(
    records: list[dict],
    n_perm: int = 2000,
    seed: int = 7,
) -> list[dict]:
    """Pairwise correlation (Pearson/Spearman + permutation p-value) between
    VLM signals and quantitative safety metrics across the evaluated sequences.

    This operationalizes RQ3: *do VLM semantic signals correlate with the
    numeric safety metrics produced by runtime verification?*
    """
    results: list[dict] = []
    for signal_name, signal_fn in VLM_SIGNALS.items():
        x = np.array([signal_fn(r) for r in records], dtype=float)
        for metric_name, metric_fn in SAFETY_METRICS.items():
            y = np.array([metric_fn(r) for r in records], dtype=float)
            x_c, y_c = _clean(x, y)
            if len(x_c) < 2:
                continue

            r_p = pearson(x_c, y_c)
            r_s = spearman(x_c, y_c)
            p_p = _permutation_p_value(x_c, y_c, r_p, n_perm=n_perm, seed=seed) if r_p is not None else None
            p_s = _permutation_p_value(x_c, y_c, r_s, n_perm=n_perm, seed=seed) if r_s is not None else None

            results.append(
                {
                    "vlm_signal": signal_name,
                    "safety_metric": metric_name,
                    "n": int(len(x_c)),
                    "pearson_r": round(r_p, 4) if r_p is not None else None,
                    "pearson_p": round(p_p, 4) if p_p is not None else None,
                    "spearman_r": round(r_s, 4) if r_s is not None else None,
                    "spearman_p": round(p_s, 4) if p_s is not None else None,
                }
            )
    return results
