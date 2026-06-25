#!/usr/bin/env python3
"""Scheme v3 model comparison for open-cluster angular-momentum evolution.

This version compares three hypotheses in the same logit(f_L)-age space:

Model A: single power law
Model B: broken power law
Model C: physical two-phase despinning curve
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from scheme_v2_analysis import (
    bic_transition_diagnostic,
    finite_logit,
    posterior_curves,
    prepare_v2_data,
    posterior_good_probabilities,
    run_numpyro_model,
    summarize_samples,
    weighted_loglike,
)


def single_powerlaw_prediction(params: np.ndarray, age: np.ndarray) -> np.ndarray:
    """Linear model in logit(f_L)-log10(age/Myr) space."""
    a, b = params[:2]
    x = np.log10(age)
    return a + b * x


def broken_powerlaw_prediction(params: np.ndarray, age: np.ndarray) -> np.ndarray:
    """Piecewise linear model in logit(f_L)-log10(age/Myr) space."""
    a, b1, b2, log_tb = params[:4]
    x = np.log10(age)
    xb = np.log10(np.exp(log_tb))
    return np.where(x < xb, a + b1 * x, a + b1 * xb + b2 * (x - xb))


def physical_two_phase_prediction(params: np.ndarray, age: np.ndarray) -> np.ndarray:
    """Scheme v2 physical two-phase curve, returned in logit(f_L) space."""
    logit_f0, log_tau, log_tb, log_beta = params[:4]
    f0 = 1.0 / (1.0 + np.exp(-logit_f0))
    tau = np.exp(log_tau)
    tb = np.exp(log_tb)
    beta = np.exp(log_beta)
    f = np.clip(f0 * np.exp(-age / tau) / (1.0 + (age / tb) ** beta), 1e-5, 1.0 - 1e-5)
    return finite_logit(f)


def fit_single_powerlaw(df: pd.DataFrame) -> dict[str, object]:
    y = df["logit_f_L"].to_numpy(float)
    age = df["age_Myr"].to_numpy(float)
    err = df["logit_f_L_err"].to_numpy(float)

    def objective(params: np.ndarray) -> float:
        yhat = single_powerlaw_prediction(params, age)
        sigma_int = params[2]
        return -weighted_loglike(y, yhat, np.sqrt(err**2 + sigma_int**2))

    result = minimize(
        objective,
        x0=np.array([-1.5, -0.2, 0.5]),
        bounds=[(-8.0, 8.0), (-5.0, 5.0), (1e-6, 5.0)],
        method="L-BFGS-B",
        options={"maxiter": 20_000},
    )
    yhat = single_powerlaw_prediction(result.x, age)
    sigma_int = float(result.x[2])
    ll = weighted_loglike(y, yhat, np.sqrt(err**2 + sigma_int**2))
    bic = 3 * np.log(len(y)) - 2 * ll
    aic = 2 * 3 - 2 * ll
    return {
        "params": result.x,
        "bic": float(bic),
        "aic": float(aic),
        "loglike": float(ll),
        "sigma_int": sigma_int,
        "success": bool(result.success),
    }


def fit_broken_powerlaw(df: pd.DataFrame) -> dict[str, object]:
    y = df["logit_f_L"].to_numpy(float)
    age = df["age_Myr"].to_numpy(float)
    err = df["logit_f_L_err"].to_numpy(float)

    def objective(params: np.ndarray) -> float:
        yhat = broken_powerlaw_prediction(params, age)
        sigma_int = params[4]
        return -weighted_loglike(y, yhat, np.sqrt(err**2 + sigma_int**2))

    result = minimize(
        objective,
        x0=np.array([-1.5, -0.2, -0.8, np.log(100.0), 0.5]),
        bounds=[(-8.0, 8.0), (-5.0, 5.0), (-5.0, 5.0), (np.log(20.0), np.log(1000.0)), (1e-6, 5.0)],
        method="L-BFGS-B",
        options={"maxiter": 40_000},
    )
    yhat = broken_powerlaw_prediction(result.x, age)
    sigma_int = float(result.x[4])
    ll = weighted_loglike(y, yhat, np.sqrt(err**2 + sigma_int**2))
    bic = 5 * np.log(len(y)) - 2 * ll
    aic = 2 * 5 - 2 * ll
    return {
        "params": result.x,
        "bic": float(bic),
        "aic": float(aic),
        "loglike": float(ll),
        "sigma_int": sigma_int,
        "success": bool(result.success),
    }


def fit_physical_two_phase(df: pd.DataFrame) -> dict[str, object]:
    y = df["logit_f_L"].to_numpy(float)
    age = df["age_Myr"].to_numpy(float)
    err = df["logit_f_L_err"].to_numpy(float)

    def objective(params: np.ndarray) -> float:
        yhat = physical_two_phase_prediction(params, age)
        sigma_int = params[4]
        return -weighted_loglike(y, yhat, np.sqrt(err**2 + sigma_int**2))

    result = minimize(
        objective,
        x0=np.array([0.0, np.log(80.0), np.log(100.0), np.log(0.5), 0.5]),
        bounds=[
            (-4.0, 4.0),
            (np.log(10.0), np.log(1000.0)),
            (np.log(20.0), np.log(1000.0)),
            (np.log(0.02), np.log(5.0)),
            (1e-6, 5.0),
        ],
        method="L-BFGS-B",
        options={"maxiter": 40_000},
    )
    yhat = physical_two_phase_prediction(result.x, age)
    sigma_int = float(result.x[4])
    ll = weighted_loglike(y, yhat, np.sqrt(err**2 + sigma_int**2))
    bic = 5 * np.log(len(y)) - 2 * ll
    aic = 2 * 5 - 2 * ll
    return {
        "params": result.x,
        "bic": float(bic),
        "aic": float(aic),
        "loglike": float(ll),
        "sigma_int": sigma_int,
        "success": bool(result.success),
    }


def model_comparison_v3(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, dict[str, object]]]:
    fits = {
        "single": fit_single_powerlaw(df),
        "broken": fit_broken_powerlaw(df),
        "two_phase": fit_physical_two_phase(df),
    }
    bic_single = fits["single"]["bic"]
    bic_broken = fits["broken"]["bic"]
    bic_two = fits["two_phase"]["bic"]
    rows = [
        {
            "model": "single",
            "k": 2,
            "k_with_sigma_int": 3,
            "bic": bic_single,
            "aic": fits["single"]["aic"],
            "delta_bic_vs_single": 0.0,
            "loglike": fits["single"]["loglike"],
            "sigma_int_logit": fits["single"]["sigma_int"],
            "success": fits["single"]["success"],
            "break_age_Myr": np.nan,
        },
        {
            "model": "broken",
            "k": 4,
            "k_with_sigma_int": 5,
            "bic": bic_broken,
            "aic": fits["broken"]["aic"],
            "delta_bic_vs_single": bic_broken - bic_single,
            "loglike": fits["broken"]["loglike"],
            "sigma_int_logit": fits["broken"]["sigma_int"],
            "success": fits["broken"]["success"],
            "break_age_Myr": float(np.exp(fits["broken"]["params"][3])),
        },
        {
            "model": "two_phase",
            "k": 4,
            "k_with_sigma_int": 5,
            "bic": bic_two,
            "aic": fits["two_phase"]["aic"],
            "delta_bic_vs_single": bic_two - bic_single,
            "loglike": fits["two_phase"]["loglike"],
            "sigma_int_logit": fits["two_phase"]["sigma_int"],
            "success": fits["two_phase"]["success"],
            "break_age_Myr": float(np.exp(fits["two_phase"]["params"][2])),
        },
    ]
    comparison = pd.DataFrame(rows)
    comparison["delta_bic_vs_best"] = comparison["bic"] - comparison["bic"].min()
    comparison["delta_broken_power"] = bic_broken - bic_single
    comparison["delta_two_power"] = bic_two - bic_single
    comparison["delta_two_broken"] = bic_two - bic_broken
    return comparison, fits


def broken_powerlaw_model(age, yerr, y=None):
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist

    a = numpyro.sample("a", dist.Normal(-1.5, 3.0))
    b1 = numpyro.sample("b1", dist.Normal(-0.2, 1.5))
    b2 = numpyro.sample("b2", dist.Normal(-0.8, 1.5))
    log_tb = numpyro.sample("log_tb", dist.Uniform(jnp.log(20.0), jnp.log(1000.0)))
    sigma_int = numpyro.sample("sigma_int", dist.HalfNormal(0.8))

    x = jnp.log10(age)
    xb = jnp.log10(jnp.exp(log_tb))
    mu = jnp.where(x < xb, a + b1 * x, a + b1 * xb + b2 * (x - xb))
    numpyro.deterministic("t_break_Myr", jnp.exp(log_tb))
    numpyro.sample("obs", dist.Normal(mu, jnp.sqrt(yerr**2 + sigma_int**2)), obs=y)


def run_broken_numpyro(df: pd.DataFrame, warmup: int, samples: int, seed: int) -> dict[str, np.ndarray]:
    import jax.numpy as jnp
    from jax import random
    from numpyro.infer import MCMC, NUTS

    kernel = NUTS(broken_powerlaw_model, target_accept_prob=0.85)
    mcmc = MCMC(kernel, num_warmup=warmup, num_samples=samples, num_chains=1, progress_bar=False)
    mcmc.run(
        random.PRNGKey(seed),
        age=jnp.asarray(df["age_Myr"].to_numpy(float)),
        yerr=jnp.asarray(df["logit_f_L_err"].to_numpy(float)),
        y=jnp.asarray(df["logit_f_L"].to_numpy(float)),
    )
    return {key: np.asarray(value) for key, value in mcmc.get_samples().items()}


def summarize_broken_samples(samples: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for name, arr in samples.items():
        values = np.asarray(arr, dtype=float)
        rows.append(
            {
                "parameter": f"broken_{name}",
                "p16": float(np.percentile(values, 16)),
                "p50": float(np.percentile(values, 50)),
                "p84": float(np.percentile(values, 84)),
                "mean": float(values.mean()),
                "sd": float(values.std(ddof=1)),
            }
        )
    return pd.DataFrame(rows).sort_values("parameter")


def prediction_curves(fits: dict[str, dict[str, object]], age_grid: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "single": 1.0 / (1.0 + np.exp(-single_powerlaw_prediction(fits["single"]["params"], age_grid))),
        "broken": 1.0 / (1.0 + np.exp(-broken_powerlaw_prediction(fits["broken"]["params"], age_grid))),
        "two_phase": 1.0 / (1.0 + np.exp(-physical_two_phase_prediction(fits["two_phase"]["params"], age_grid))),
    }


def make_figure1_population(df: pd.DataFrame, fits: dict[str, dict[str, object]], outdir: Path) -> None:
    age_grid = np.logspace(np.log10(df["age_Myr"].min() * 0.8), np.log10(df["age_Myr"].max() * 1.2), 300)
    curves = prediction_curves(fits, age_grid)
    clean = df["p_good_v3"] > 0.7

    fig, ax = plt.subplots(figsize=(7.5, 5.4))
    sc = ax.scatter(
        df.loc[clean, "age_Myr"],
        df.loc[clean, "f_L_v2"],
        c=df.loc[clean, "R_GC_kpc"],
        s=34,
        cmap="viridis",
        edgecolor="none",
        label=r"$P({\rm good})>0.7$",
    )
    ax.scatter(df.loc[~clean, "age_Myr"], df.loc[~clean, "f_L_v2"], c="0.78", s=18, edgecolor="none", label="mixture downweighted")
    ax.plot(age_grid, curves["single"], color="0.15", lw=1.4, ls=":", label="single")
    ax.plot(age_grid, curves["broken"], color="darkorange", lw=2.0, label="broken")
    ax.plot(age_grid, curves["two_phase"], color="steelblue", lw=2.0, label="two-phase")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Age (Myr)")
    ax.set_ylabel(r"Bounded $f_L$")
    ax.legend(frameon=False, fontsize=9)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(r"$R_{\rm GC}$ (kpc)")
    fig.tight_layout()
    fig.savefig(outdir / "figure1_population.png", dpi=240)
    plt.close(fig)


def make_figure2_model_comparison(comparison: pd.DataFrame, outdir: Path) -> None:
    order = ["single", "broken", "two_phase"]
    labels = ["Single", "Broken", "Two-phase"]
    values = [float(comparison.loc[comparison["model"] == name, "delta_bic_vs_single"].iloc[0]) for name in order]
    colors = ["0.5", "darkorange", "steelblue"]

    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    ax.bar(labels, values, color=colors)
    ax.axhline(0, color="black", lw=1.0)
    ax.set_ylabel(r"$\Delta{\rm BIC}$ relative to single")
    ax.set_title("Model comparison")
    for i, value in enumerate(values):
        va = "bottom" if value >= 0 else "top"
        offset = 0.03 * (max(abs(v) for v in values) or 1.0)
        ax.text(i, value + (offset if value >= 0 else -offset), f"{value:.1f}", ha="center", va=va)
    fig.tight_layout()
    fig.savefig(outdir / "figure2_model_comparison.png", dpi=240)
    plt.close(fig)


def make_figure3_break_age_posterior(
    broken_samples: dict[str, np.ndarray],
    two_phase_samples: dict[str, np.ndarray],
    outdir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    ax.hist(broken_samples["t_break_Myr"], bins=32, alpha=0.62, color="darkorange", label="broken")
    ax.hist(two_phase_samples["t_break_Myr"], bins=32, alpha=0.52, color="steelblue", label="two-phase")
    ax.set_xlabel(r"Break / transition age $t_b$ (Myr)")
    ax.set_ylabel("Posterior samples")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outdir / "figure3_break_age_posterior.png", dpi=240)
    plt.close(fig)


def make_scheme_v3_panel_d_figure(
    df: pd.DataFrame,
    two_phase_samples: dict[str, np.ndarray],
    comparison: pd.DataFrame,
    outdir: Path,
) -> None:
    age_grid = np.logspace(np.log10(df["age_Myr"].min() * 0.8), np.log10(df["age_Myr"].max() * 1.2), 240)
    curves = posterior_curves(two_phase_samples, age_grid)
    q16, q50, q84 = np.percentile(curves, [16, 50, 84], axis=0)
    clean = df["p_good_v3"] > 0.7

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), constrained_layout=True)
    ax = axes[0, 0]
    ax.scatter(df["age_Myr"], df["f_L_raw"], s=24, c="0.55", alpha=0.75, edgecolor="none")
    ax.axhline(1.0, color="firebrick", lw=1.0, ls="--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Age (Myr)")
    ax.set_ylabel(r"Raw $J_{\rm int}/J_{\rm virial}$")
    ax.set_title("A. Raw diagnostic")

    ax = axes[0, 1]
    sc = ax.scatter(df.loc[clean, "age_Myr"], df.loc[clean, "f_L_v2"], c=df.loc[clean, "R_GC_kpc"], s=34, cmap="viridis", edgecolor="none")
    ax.scatter(df.loc[~clean, "age_Myr"], df.loc[~clean, "f_L_v2"], s=18, c="0.8", edgecolor="none", alpha=0.65)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Age (Myr)")
    ax.set_ylabel(r"Bounded $f_L$")
    ax.set_title(r"B. Cleaned population, $P({\rm good})>0.7$")
    fig.colorbar(sc, ax=ax, label=r"$R_{\rm GC}$ (kpc)")

    ax = axes[1, 0]
    ax.fill_between(age_grid, q16, q84, color="steelblue", alpha=0.24, lw=0)
    ax.plot(age_grid, q50, color="steelblue", lw=2.0)
    ax.scatter(df["age_Myr"], df["f_L_v2"], s=18, c=df["p_good_v3"], cmap="magma", edgecolor="none")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Age (Myr)")
    ax.set_ylabel(r"$f_L$")
    ax.set_title("C. Two-phase posterior predictive")

    ax = axes[1, 1]
    order = ["single", "broken", "two_phase"]
    labels = ["Single", "Broken", "Two-phase"]
    values = [float(comparison.loc[comparison["model"] == name, "delta_bic_vs_single"].iloc[0]) for name in order]
    ax.bar(labels, values, color=["0.5", "darkorange", "steelblue"])
    ax.axhline(0, color="black", lw=1.0)
    ax.set_ylabel(r"$\Delta{\rm BIC}$ relative to single")
    ax.set_title("D. Model comparison")
    fig.savefig(outdir / "figure1_scheme_v3_four_panel.png", dpi=240)
    plt.close(fig)


def write_summary(
    df: pd.DataFrame,
    comparison: pd.DataFrame,
    posterior: pd.DataFrame,
    outdir: Path,
) -> None:
    best = comparison.sort_values("bic").iloc[0]
    broken = comparison.loc[comparison["model"] == "broken"].iloc[0]
    two = comparison.loc[comparison["model"] == "two_phase"].iloc[0]
    single = comparison.loc[comparison["model"] == "single"].iloc[0]

    def p50(param: str) -> float:
        hit = posterior.loc[posterior["parameter"] == param]
        return float(hit["p50"].iloc[0]) if len(hit) else np.nan

    if best["model"] == "two_phase" and two["delta_bic_vs_single"] < -10:
        verdict = "Two-phase physical despinning is favored over the single-power-law diagnostic."
    elif best["model"] == "broken" and broken["delta_bic_vs_single"] < -10:
        verdict = "The data favor a statistical break/characteristic age scale, but not the stronger physical two-phase model."
    else:
        verdict = "The current Gaia-only measurements do not justify a break or physical two-phase claim over the single-power-law diagnostic."

    lines = [
        "# Scheme v3 Model Comparison Summary",
        "",
        f"Input fitted clusters: {len(df)}",
        f"Cleaned population P(good)>0.7: {int((df['p_good_v3'] > 0.7).sum())}",
        f"Best BIC model: {best['model']}",
        "",
        "## BIC comparison",
        "",
        f"Single: BIC = {single['bic']:.3f}, Delta BIC = {single['delta_bic_vs_single']:.3f}",
        f"Broken: BIC = {broken['bic']:.3f}, Delta BIC = {broken['delta_bic_vs_single']:.3f}, break = {broken['break_age_Myr']:.2f} Myr",
        f"Two-phase: BIC = {two['bic']:.3f}, Delta BIC = {two['delta_bic_vs_single']:.3f}, transition = {two['break_age_Myr']:.2f} Myr",
        f"Delta BIC(two-phase - broken) = {two['delta_two_broken']:.3f}",
        "",
        "## Bayesian break-age summaries",
        "",
        f"Broken-power-law posterior t_b median = {p50('broken_t_break_Myr'):.2f} Myr",
        f"Two-phase posterior t_b median = {p50('t_break_Myr'):.2f} Myr",
        "",
        "## Verdict",
        "",
        verdict,
    ]
    (outdir / "summary.md").write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("outputs/cluster_spin_results.csv"))
    parser.add_argument("--clusters", type=Path, default=Path("hunt24/clusters.dat"))
    parser.add_argument("--outdir", type=Path, default=Path("outputs_v3"))
    parser.add_argument("--warmup", type=int, default=800)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260619)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    df = prepare_v2_data(args.results, args.clusters)
    two_phase_samples = run_numpyro_model(df, warmup=args.warmup, samples=args.samples, seed=args.seed)
    broken_samples = run_broken_numpyro(df, warmup=args.warmup, samples=args.samples, seed=args.seed + 17)
    df["p_good_v3"] = posterior_good_probabilities(df, two_phase_samples)

    comparison, fits = model_comparison_v3(df)
    two_phase_summary = summarize_samples(two_phase_samples)
    broken_summary = summarize_broken_samples(broken_samples)
    posterior = pd.concat([two_phase_summary, broken_summary], ignore_index=True)

    df.to_csv(args.outdir / "cluster_population.csv", index=False)
    posterior.to_csv(args.outdir / "posterior_summary.csv", index=False)
    comparison.to_csv(args.outdir / "model_comparison.csv", index=False)

    make_figure1_population(df, fits, args.outdir)
    make_figure2_model_comparison(comparison, args.outdir)
    make_figure3_break_age_posterior(broken_samples, two_phase_samples, args.outdir)
    make_scheme_v3_panel_d_figure(df, two_phase_samples, comparison, args.outdir)
    write_summary(df, comparison, posterior, args.outdir)

    legacy = bic_transition_diagnostic(df)
    pd.DataFrame([legacy]).to_csv(args.outdir / "legacy_v2_two_phase_vs_power.csv", index=False)

    best = comparison.sort_values("bic").iloc[0]
    broken = comparison.loc[comparison["model"] == "broken"].iloc[0]
    two = comparison.loc[comparison["model"] == "two_phase"].iloc[0]
    print(f"Scheme v3 fitted clusters: {len(df)}")
    print(f"Best BIC model: {best['model']}")
    print(f"Broken Delta BIC vs single: {broken['delta_bic_vs_single']:.3f}")
    print(f"Two-phase Delta BIC vs single: {two['delta_bic_vs_single']:.3f}")
    print(f"Two-phase Delta BIC vs broken: {two['delta_two_broken']:.3f}")
    print(f"Wrote outputs to {args.outdir}")


if __name__ == "__main__":
    main()
