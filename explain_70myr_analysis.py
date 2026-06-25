#!/usr/bin/env python3
"""Follow-up checks for the fitted broken-model age from scheme_v3.

The script runs the three tests requested in ``explain_70myr.txt``:

1. Broken-power-law break age versus Galactocentric environment.
2. Broken-power-law break age versus cluster mass.
3. Comparison between the break age and classical relaxation time.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from scheme_v3_analysis import fit_broken_powerlaw


PC_PER_KMS_TO_MYR = 0.9777922216807892


def load_population(path: Path, min_good_probability: float) -> pd.DataFrame:
    df = pd.read_csv(path)
    ok = (
        np.isfinite(df["age_Myr"])
        & np.isfinite(df["f_L_v2"])
        & np.isfinite(df["logit_f_L"])
        & np.isfinite(df["logit_f_L_err"])
        & np.isfinite(df["R_GC_kpc"])
        & np.isfinite(df["MassJ_Msun"])
        & np.isfinite(df["r_h_pc"])
        & np.isfinite(df["sigma1d_kms"])
        & np.isfinite(df["N_catalog"])
    )
    if "p_good_v3" in df.columns:
        ok &= df["p_good_v3"] >= min_good_probability
    df = df.loc[ok].copy().reset_index(drop=True)
    df["t_cross_Myr"] = df["r_h_pc"] / df["sigma1d_kms"] * PC_PER_KMS_TO_MYR
    safe_n = np.maximum(df["N_catalog"].to_numpy(float), 3.0)
    df["t_relax_Myr"] = 0.1 * safe_n / np.log(safe_n) * df["t_cross_Myr"]
    df["age_over_t_relax"] = df["age_Myr"] / df["t_relax_Myr"]
    return df


def fit_group_breaks(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows = []
    for group, sub in df.groupby(group_col, observed=True):
        if len(sub) < 8:
            rows.append(
                {
                    "experiment": group_col,
                    "group": str(group),
                    "n": len(sub),
                    "t_break_Myr": np.nan,
                    "bic": np.nan,
                    "loglike": np.nan,
                    "status": "too_few_clusters",
                }
            )
            continue
        try:
            fit = fit_broken_powerlaw(sub)
            params = fit["params"]
            rows.append(
                {
                    "experiment": group_col,
                    "group": str(group),
                    "n": len(sub),
                    "t_break_Myr": float(np.exp(params[3])),
                    "b1": float(params[1]),
                    "b2": float(params[2]),
                    "bic": float(fit["bic"]),
                    "loglike": float(fit["loglike"]),
                    "status": "ok" if fit["success"] else "optimizer_warning",
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "experiment": group_col,
                    "group": str(group),
                    "n": len(sub),
                    "t_break_Myr": np.nan,
                    "bic": np.nan,
                    "loglike": np.nan,
                    "status": f"failed:{type(exc).__name__}",
                }
            )
    return pd.DataFrame(rows)


def add_group_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["R_GC_region"] = pd.cut(
        out["R_GC_kpc"],
        bins=[-np.inf, 8.0, 10.0, np.inf],
        labels=["inner_Rgc_lt_8", "solar_8_to_10", "outer_Rgc_gt_10"],
    )
    out["mass_bin"] = pd.cut(
        out["MassJ_Msun"],
        bins=[-np.inf, 500.0, 2000.0, np.inf],
        labels=["low_M_lt_500", "mid_M_500_to_2000", "high_M_gt_2000"],
    )
    return out


def fit_mass_scaling(group_breaks: pd.DataFrame, df: pd.DataFrame) -> dict[str, float]:
    ok = group_breaks["status"].eq("ok") & np.isfinite(group_breaks["t_break_Myr"])
    gb = group_breaks.loc[ok].copy()
    if len(gb) < 3:
        return {"gamma": np.nan, "amplitude_at_1000Msun": np.nan, "n_bins": int(len(gb))}

    med_mass = []
    for group in gb["group"]:
        sub = df[df["mass_bin"].astype(str) == group]
        med_mass.append(float(sub["MassJ_Msun"].median()))
    x = np.log10(np.asarray(med_mass) / 1000.0)
    y = np.log10(gb["t_break_Myr"].to_numpy(float))
    slope, intercept = np.polyfit(x, y, deg=1)
    return {
        "gamma": float(slope),
        "amplitude_at_1000Msun": float(10**intercept),
        "n_bins": int(len(gb)),
    }


def relaxation_summary(df: pd.DataFrame, break_age_myr: float) -> dict[str, float]:
    ratio = break_age_myr / df["t_relax_Myr"].to_numpy(float)
    finite = ratio[np.isfinite(ratio) & (ratio > 0)]
    return {
        "break_age_Myr": float(break_age_myr),
        "median_t_relax_Myr": float(df["t_relax_Myr"].median()),
        "median_t_break_over_t_relax": float(np.median(finite)),
        "p16_t_break_over_t_relax": float(np.percentile(finite, 16)),
        "p84_t_break_over_t_relax": float(np.percentile(finite, 84)),
        "fraction_between_1_and_3_trelax": float(np.mean((finite >= 1.0) & (finite <= 3.0))),
    }


def robust_break_age(model_comparison_path: Path, fallback: float = 70.0) -> float:
    if not model_comparison_path.exists():
        return fallback
    comparison = pd.read_csv(model_comparison_path)
    hit = comparison[comparison["model"] == "broken"]
    if hit.empty or not np.isfinite(hit["break_age_Myr"].iloc[0]):
        return fallback
    return float(hit["break_age_Myr"].iloc[0])


def make_environment_figure(group_breaks: pd.DataFrame, outdir: Path) -> None:
    rows = group_breaks[group_breaks["experiment"] == "R_GC_region"].copy()
    order = ["inner_Rgc_lt_8", "solar_8_to_10", "outer_Rgc_gt_10"]
    rows["group"] = pd.Categorical(rows["group"], categories=order, ordered=True)
    rows = rows.sort_values("group")

    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    x = np.arange(len(rows))
    ax.bar(x, rows["t_break_Myr"], color=["#8c4b35", "#4f7f9f", "#6f8f4e"])
    for i, row in enumerate(rows.itertuples()):
        ax.text(i, row.t_break_Myr, f"n={row.n}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel(r"Broken-power-law $t_b$ (Myr)")
    ax.set_xlabel(r"Galactocentric region")
    label_map = {
        "inner_Rgc_lt_8": "Inner\nR<8",
        "solar_8_to_10": "Solar\n8<R<10",
        "outer_Rgc_gt_10": "Outer\nR>10",
    }
    ax.set_xticks(x)
    ax.set_xticklabels([label_map.get(str(group), str(group)) for group in rows["group"].astype(str)])
    fig.tight_layout()
    fig.savefig(outdir / "figure_environment_breaks.png", dpi=240)
    plt.close(fig)


def make_mass_figure(group_breaks: pd.DataFrame, df: pd.DataFrame, outdir: Path) -> None:
    rows = group_breaks[group_breaks["experiment"] == "mass_bin"].copy()
    order = ["low_M_lt_500", "mid_M_500_to_2000", "high_M_gt_2000"]
    rows["group"] = pd.Categorical(rows["group"], categories=order, ordered=True)
    rows = rows.sort_values("group")
    med_mass = []
    for group in rows["group"].astype(str):
        med_mass.append(float(df[df["mass_bin"].astype(str) == group]["MassJ_Msun"].median()))
    rows["median_mass"] = med_mass

    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    ax.plot(rows["median_mass"], rows["t_break_Myr"], marker="o", color="#4c5f8f", lw=1.8)
    for row in rows.itertuples():
        ax.text(row.median_mass, row.t_break_Myr, f" n={row.n}", va="center", fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel(r"Median bin mass ($M_\odot$)")
    ax.set_ylabel(r"Broken-power-law $t_b$ (Myr)")
    fig.tight_layout()
    fig.savefig(outdir / "figure_mass_breaks.png", dpi=240)
    plt.close(fig)


def make_relaxation_figure(df: pd.DataFrame, break_age_myr: float, outdir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)

    ax = axes[0]
    sc = ax.scatter(df["t_relax_Myr"], df["age_Myr"], c=df["f_L_v2"], s=30, cmap="magma", edgecolor="none")
    xgrid = np.logspace(np.log10(df["t_relax_Myr"].min() * 0.7), np.log10(df["t_relax_Myr"].max() * 1.3), 120)
    ax.plot(xgrid, xgrid, color="0.25", lw=1.0, ls="--", label=r"$t=t_{\rm relax}$")
    ax.plot(xgrid, 3 * xgrid, color="0.45", lw=1.0, ls=":", label=r"$t=3t_{\rm relax}$")
    ax.axhline(break_age_myr, color="firebrick", lw=1.5, label=fr"$t_b={break_age_myr:.1f}$ Myr")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$t_{\rm relax}$ (Myr)")
    ax.set_ylabel("Cluster age (Myr)")
    ax.legend(frameon=False, fontsize=8)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(r"$f_L$")

    ax = axes[1]
    ratio = break_age_myr / df["t_relax_Myr"]
    ax.hist(ratio[np.isfinite(ratio)], bins=24, color="#607d8b", alpha=0.88)
    ax.axvspan(1.0, 3.0, color="darkorange", alpha=0.18, label=r"$1-3\,t_{\rm relax}$")
    ax.axvline(np.nanmedian(ratio), color="firebrick", lw=1.8, label="median")
    ax.set_xlabel(r"$t_b/t_{\rm relax}$")
    ax.set_ylabel("Clusters")
    ax.legend(frameon=False, fontsize=8)

    fig.savefig(outdir / "figure_relaxation_time_test.png", dpi=240)
    plt.close(fig)


def make_summary_table(df: pd.DataFrame, outdir: Path) -> None:
    cols = [
        "Name",
        "age_Myr",
        "MassJ_Msun",
        "R_GC_kpc",
        "r_h_pc",
        "sigma1d_kms",
        "N_catalog",
        "t_cross_Myr",
        "t_relax_Myr",
        "age_over_t_relax",
        "f_L_v2",
        "p_good_v3",
    ]
    df[cols].sort_values("age_Myr").to_csv(outdir / "cluster_relaxation_times.csv", index=False)


def write_summary(
    group_breaks: pd.DataFrame,
    mass_scaling: dict[str, float],
    relax: dict[str, float],
    outdir: Path,
) -> None:
    def section_rows(experiment: str) -> list[str]:
        rows = group_breaks[group_breaks["experiment"] == experiment]
        lines = []
        for row in rows.itertuples():
            if np.isfinite(row.t_break_Myr):
                lines.append(f"- {row.group}: t_b = {row.t_break_Myr:.2f} Myr (n={row.n}, status={row.status})")
            else:
                lines.append(f"- {row.group}: unavailable (n={row.n}, status={row.status})")
        return lines

    if relax["fraction_between_1_and_3_trelax"] >= 0.5:
        verdict = "The 70 Myr scale is plausibly linked to relaxation: most clusters place the break within 1-3 relaxation times."
    else:
        verdict = "The current sample does not support a simple t_b ~ 1-3 t_relax interpretation."

    lines = [
        "# Follow-up Checks for the Fitted Broken-Model Age",
        "",
        "## Experiment 1: Galactocentric environment",
        "",
        *section_rows("R_GC_region"),
        "",
        "## Experiment 2: Mass dependence",
        "",
        *section_rows("mass_bin"),
        "",
        f"Mass scaling from three bins: t_b = {mass_scaling['amplitude_at_1000Msun']:.2f} Myr * (M/1000 Msun)^{mass_scaling['gamma']:.3f}",
        "",
        "## Experiment 3: Relaxation-time comparison",
        "",
        f"Reference break age: {relax['break_age_Myr']:.2f} Myr",
        f"Median t_relax: {relax['median_t_relax_Myr']:.2f} Myr",
        f"Median t_b/t_relax: {relax['median_t_break_over_t_relax']:.2f}",
        f"16-84% t_b/t_relax range: {relax['p16_t_break_over_t_relax']:.2f} - {relax['p84_t_break_over_t_relax']:.2f}",
        f"Fraction with 1 <= t_b/t_relax <= 3: {relax['fraction_between_1_and_3_trelax']:.3f}",
        "",
        "## Verdict",
        "",
        verdict,
    ]
    (outdir / "summary.md").write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population", type=Path, default=Path("outputs_v3/cluster_population.csv"))
    parser.add_argument("--model-comparison", type=Path, default=Path("outputs_v3/model_comparison.csv"))
    parser.add_argument("--outdir", type=Path, default=Path("outputs_70myr"))
    parser.add_argument("--min-good-probability", type=float, default=0.7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    df = load_population(args.population, args.min_good_probability)
    df = add_group_columns(df)
    break_age = robust_break_age(args.model_comparison)

    env_breaks = fit_group_breaks(df, "R_GC_region")
    mass_breaks = fit_group_breaks(df, "mass_bin")
    group_breaks = pd.concat([env_breaks, mass_breaks], ignore_index=True)
    mass_scaling = fit_mass_scaling(mass_breaks, df)
    relax = relaxation_summary(df, break_age)

    group_breaks.to_csv(args.outdir / "group_breaks.csv", index=False)
    pd.DataFrame([mass_scaling]).to_csv(args.outdir / "mass_scaling.csv", index=False)
    pd.DataFrame([relax]).to_csv(args.outdir / "relaxation_summary.csv", index=False)
    make_summary_table(df, args.outdir)
    make_environment_figure(group_breaks, args.outdir)
    make_mass_figure(group_breaks, df, args.outdir)
    make_relaxation_figure(df, break_age, args.outdir)
    write_summary(group_breaks, mass_scaling, relax, args.outdir)

    print(f"Clean clusters used: {len(df)}")
    print(f"Reference broken-power-law break: {break_age:.2f} Myr")
    print(f"Median t_b/t_relax: {relax['median_t_break_over_t_relax']:.2f}")
    print(f"Fraction within 1-3 t_relax: {relax['fraction_between_1_and_3_trelax']:.3f}")
    print(f"Wrote outputs to {args.outdir}")


if __name__ == "__main__":
    main()
