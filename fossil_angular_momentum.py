#!/usr/bin/env python3
"""Measure fossil angular momentum diagnostics for Hunt+2024 open clusters.

The script is intentionally self-contained: it parses the local CDS fixed-width
tables, builds a quality-controlled cluster sample, fits a 3D velocity-gradient
tensor for each cluster with enough Gaia DR3 radial velocities, and writes the
first-pass tables and figures needed for the research scheme.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord


KMS_PER_PC_TO_MYR_INV = 1.0227121650537077
G_PC_MSUN_KMS = 4.30091727003628e-3


CLUSTER_SPECS = {
    "Name": (0, 20),
    "ID": (21, 25),
    "Type": (280, 281),
    "N": (294, 300),
    "Nt": (313, 318),
    "RAdeg": (319, 331),
    "DEdeg": (332, 344),
    "GLON": (345, 357),
    "GLAT": (358, 369),
    "r50pc": (418, 431),
    "rtpc": (446, 459),
    "dist50": (599, 614),
    "pmRA": (474, 487),
    "pmDE": (511, 523),
    "RV": (691, 704),
    "s_RV": (705, 718),
    "n_RV": (733, 737),
    "CMDCl50": (758, 768),
    "CMDClHuman": (791, 794),
    "logAge16": (795, 806),
    "logAge50": (807, 818),
    "logAge84": (819, 831),
    "r50Jpc": (964, 976),
    "rJpc": (977, 990),
    "probJ": (991, 1001),
    "NJ": (1002, 1006),
    "MassJ": (1007, 1022),
    "e_MassJ": (1023, 1038),
    "MassTot": (1039, 1054),
    "e_MassTot": (1055, 1070),
    "Note": (1098, 1131),
}


MEMBER_SPECS = {
    "Seq": (0, 7),
    "Name": (8, 28),
    "ID": (29, 33),
    "GaiaDR3": (34, 53),
    "inrj": (54, 55),
    "inrt": (56, 57),
    "Prob": (58, 78),
    "RAdeg": (79, 103),
    "DEdeg": (126, 148),
    "pmRA": (220, 243),
    "e_pmRA": (244, 265),
    "pmDE": (266, 289),
    "e_pmDE": (290, 310),
    "Plx": (311, 334),
    "e_Plx": (335, 355),
    "RUWE": (588, 607),
    "FidelityV1": (608, 626),
    "Gmag": (788, 807),
    "BP_RP": (848, 871),
    "RV": (920, 942),
    "e_RV": (943, 963),
    "GRVSmag": (979, 998),
    "NSS": (1089, 1090),
    "RVS": (1091, 1092),
    "Mass50": (1135, 1155),
}


NUMERIC_CLUSTER_COLUMNS = [
    "ID",
    "N",
    "Nt",
    "RAdeg",
    "DEdeg",
    "GLON",
    "GLAT",
    "r50pc",
    "rtpc",
    "dist50",
    "pmRA",
    "pmDE",
    "RV",
    "s_RV",
    "n_RV",
    "CMDCl50",
    "logAge16",
    "logAge50",
    "logAge84",
    "r50Jpc",
    "rJpc",
    "probJ",
    "NJ",
    "MassJ",
    "e_MassJ",
    "MassTot",
    "e_MassTot",
]

NUMERIC_MEMBER_COLUMNS = [
    "Seq",
    "ID",
    "inrj",
    "inrt",
    "Prob",
    "RAdeg",
    "DEdeg",
    "pmRA",
    "e_pmRA",
    "pmDE",
    "e_pmDE",
    "Plx",
    "e_Plx",
    "RUWE",
    "FidelityV1",
    "Gmag",
    "BP_RP",
    "RV",
    "e_RV",
    "GRVSmag",
    "NSS",
    "RVS",
    "Mass50",
]


def read_fwf(path: Path, specs: dict[str, tuple[int, int]], numeric: list[str]) -> pd.DataFrame:
    names = list(specs)
    colspecs = [specs[name] for name in names]
    df = pd.read_fwf(path, colspecs=colspecs, names=names, dtype=str, keep_default_na=False)
    for col in df.columns:
        df[col] = df[col].astype(str).str.strip()
    for col in numeric:
        values = df[col].mask(df[col].isin(["", "?"]), np.nan)
        df[col] = pd.to_numeric(values, errors="coerce")
    return df


def read_member_chunks(
    path: Path,
    specs: dict[str, tuple[int, int]],
    numeric: list[str],
    ids: set[int],
    chunksize: int = 150_000,
) -> pd.DataFrame:
    names = list(specs)
    colspecs = [specs[name] for name in names]
    chunks = []
    for chunk in pd.read_fwf(
        path,
        colspecs=colspecs,
        names=names,
        dtype=str,
        keep_default_na=False,
        chunksize=chunksize,
    ):
        for col in chunk.columns:
            chunk[col] = chunk[col].astype(str).str.strip()
        chunk["ID"] = pd.to_numeric(chunk["ID"].replace({"": np.nan}), errors="coerce")
        chunk = chunk[chunk["ID"].isin(ids)].copy()
        if chunk.empty:
            continue
        for col in numeric:
            if col == "ID":
                continue
            values = chunk[col].mask(chunk[col].isin(["", "?"]), np.nan)
            chunk[col] = pd.to_numeric(values, errors="coerce")
        chunk["ID"] = chunk["ID"].astype(int)
        chunks.append(chunk)
    if not chunks:
        return pd.DataFrame(columns=names)
    return pd.concat(chunks, ignore_index=True)


def select_clusters(
    clusters: pd.DataFrame,
    min_members: int,
    min_catalog_rv: int,
    min_log_age: float,
    max_log_age: float,
    min_probj: float,
) -> pd.DataFrame:
    cmd_human = clusters["CMDClHuman"].fillna("").astype(str)
    note = clusters["Note"].fillna("").astype(str)
    mask = (
        (clusters["Type"] == "o")
        & (clusters["N"] >= min_members)
        & (clusters["n_RV"] >= min_catalog_rv)
        & clusters["logAge50"].between(min_log_age, max_log_age, inclusive="both")
        & clusters["MassJ"].notna()
        & clusters["rJpc"].notna()
        & (clusters["probJ"] >= min_probj)
        & ~cmd_human.str.startswith("FP")
        & (note == "")
    )
    sample = clusters.loc[mask].copy().sort_values(["logAge50", "n_RV"], ascending=[True, False])
    sample["age_Myr"] = 10 ** (sample["logAge50"] - 6.0)
    return sample.reset_index(drop=True)


def skycoord_from_table(df: pd.DataFrame, dist_pc: np.ndarray, rv_col: str = "RV") -> SkyCoord:
    return SkyCoord(
        ra=df["RAdeg"].to_numpy(float) * u.deg,
        dec=df["DEdeg"].to_numpy(float) * u.deg,
        distance=dist_pc * u.pc,
        pm_ra_cosdec=df["pmRA"].to_numpy(float) * u.mas / u.yr,
        pm_dec=df["pmDE"].to_numpy(float) * u.mas / u.yr,
        radial_velocity=df[rv_col].to_numpy(float) * u.km / u.s,
        frame="icrs",
    )


def center_coord(cluster: pd.Series) -> SkyCoord:
    return SkyCoord(
        ra=float(cluster.RAdeg) * u.deg,
        dec=float(cluster.DEdeg) * u.deg,
        distance=float(cluster.dist50) * u.pc,
        pm_ra_cosdec=float(cluster.pmRA) * u.mas / u.yr,
        pm_dec=float(cluster.pmDE) * u.mas / u.yr,
        radial_velocity=float(cluster.RV) * u.km / u.s,
        frame="icrs",
    )


def finite_quality_members(members: pd.DataFrame, min_prob: float) -> pd.DataFrame:
    m = members.copy()
    base = (
        (m["Prob"] >= min_prob)
        & (m["inrj"] == 1)
        & m["RAdeg"].notna()
        & m["DEdeg"].notna()
        & m["pmRA"].notna()
        & m["pmDE"].notna()
        & m["Plx"].notna()
        & (m["Plx"] > 0)
    )
    return m.loc[base].copy()


def distances_from_parallax(
    members: pd.DataFrame,
    cluster_distance_pc: float,
    use_star_parallax: bool,
) -> np.ndarray:
    if not use_star_parallax:
        return np.full(len(members), cluster_distance_pc, dtype=float)
    plx = members["Plx"].to_numpy(float)
    distance = np.full(len(members), cluster_distance_pc, dtype=float)
    ok = np.isfinite(plx) & (plx > 0)
    distance[ok] = 1000.0 / plx[ok]
    # Keep catastrophic parallax excursions from dominating a cluster-scale fit.
    lo = max(1.0, 0.5 * cluster_distance_pc)
    hi = 1.5 * cluster_distance_pc
    return np.clip(distance, lo, hi)


def positions_relative_to_center(
    members: pd.DataFrame,
    cluster: pd.Series,
    use_star_parallax: bool,
) -> np.ndarray:
    dist_pc = distances_from_parallax(members, float(cluster.dist50), use_star_parallax)
    sc = skycoord_from_table(members.assign(RV=0.0), dist_pc, rv_col="RV")
    cc = center_coord(cluster)
    pos = sc.cartesian.xyz.to_value(u.pc).T
    cpos = cc.cartesian.xyz.to_value(u.pc)
    rel = pos - cpos
    return rel - np.nanmedian(rel, axis=0)


def velocities_relative_to_center(
    rv_members: pd.DataFrame,
    cluster: pd.Series,
    use_star_parallax: bool,
) -> tuple[np.ndarray, np.ndarray]:
    dist_pc = distances_from_parallax(rv_members, float(cluster.dist50), use_star_parallax)
    sc = skycoord_from_table(rv_members, dist_pc)
    cc = center_coord(cluster)
    pos = sc.cartesian.xyz.to_value(u.pc).T
    vel = sc.velocity.d_xyz.to_value(u.km / u.s).T
    cpos = cc.cartesian.xyz.to_value(u.pc)
    cvel = cc.velocity.d_xyz.to_value(u.km / u.s)
    rel_pos = pos - cpos
    rel_vel = vel - cvel
    rel_pos = rel_pos - np.nanmedian(rel_pos, axis=0)
    rel_vel = rel_vel - np.nanmedian(rel_vel, axis=0)
    return rel_pos, rel_vel


def fit_velocity_gradient(pos_pc: np.ndarray, vel_kms: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.column_stack([np.ones(len(pos_pc)), pos_pc])
    coeff, *_ = np.linalg.lstsq(x, vel_kms, rcond=None)
    intercept = coeff[0]
    gradient = coeff[1:].T
    fitted = x @ coeff
    residual = vel_kms - fitted
    return intercept, gradient, residual


def omega_from_gradient(gradient: np.ndarray) -> np.ndarray:
    curl = np.array(
        [
            gradient[2, 1] - gradient[1, 2],
            gradient[0, 2] - gradient[2, 0],
            gradient[1, 0] - gradient[0, 1],
        ]
    )
    return 0.5 * curl


def inertia_tensor_specific(pos_pc: np.ndarray, masses: np.ndarray) -> np.ndarray:
    masses = np.where(np.isfinite(masses) & (masses > 0), masses, np.nanmedian(masses[np.isfinite(masses)]))
    if not np.isfinite(masses).any() or np.nansum(masses) <= 0:
        masses = np.ones(len(pos_pc))
    weights = masses / np.nansum(masses)
    r2 = np.sum(pos_pc**2, axis=1)
    tensor = np.zeros((3, 3), dtype=float)
    for w, r2_i, r_i in zip(weights, r2, pos_pc):
        tensor += w * (r2_i * np.eye(3) - np.outer(r_i, r_i))
    return tensor


def sigma3d(residual_vel: np.ndarray) -> float:
    if len(residual_vel) < 2:
        return np.nan
    cov = np.cov(residual_vel.T)
    value = math.sqrt(max(0.0, float(np.trace(cov))))
    return value


def spin_axis_galactic(axis_icrs: np.ndarray) -> tuple[float, float]:
    norm = np.linalg.norm(axis_icrs)
    if norm == 0 or not np.isfinite(norm):
        return np.nan, np.nan
    axis = axis_icrs / norm
    dec = math.degrees(math.asin(float(axis[2])))
    ra = math.degrees(math.atan2(float(axis[1]), float(axis[0]))) % 360.0
    coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
    return float(coord.galactic.l.deg), float(coord.galactic.b.deg)


def diagnostic_values(
    cluster: pd.Series,
    all_pos: np.ndarray,
    masses: np.ndarray,
    gradient: np.ndarray,
    residual: np.ndarray,
) -> dict[str, float]:
    omega = omega_from_gradient(gradient)
    omega_myr = omega * KMS_PER_PC_TO_MYR_INV
    omega_mag = float(np.linalg.norm(omega))
    axis = omega / omega_mag if omega_mag > 0 else np.array([np.nan, np.nan, np.nan])
    l_gal, b_gal = spin_axis_galactic(axis)

    i_specific = inertia_tensor_specific(all_pos, masses)
    mass = float(cluster.MassJ)
    i_tensor = mass * i_specific
    l_vec = i_tensor @ omega
    l_obs = float(np.linalg.norm(l_vec))
    sig3 = sigma3d(residual)
    radius = float(cluster.rJpc)
    l_vir = mass * radius * sig3 if sig3 > 0 else np.nan
    f_l = l_obs / l_vir if l_vir and np.isfinite(l_vir) and l_vir > 0 else np.nan
    lambda_oc = l_obs / math.sqrt(2.0 * G_PC_MSUN_KMS * mass**3 * radius)

    sym = 0.5 * (gradient + gradient.T)
    asym = 0.5 * (gradient - gradient.T)
    return {
        "omega_x_kms_pc": omega[0],
        "omega_y_kms_pc": omega[1],
        "omega_z_kms_pc": omega[2],
        "omega_mag_kms_pc": omega_mag,
        "omega_mag_Myr_inv": float(np.linalg.norm(omega_myr)),
        "spin_axis_icrs_x": axis[0],
        "spin_axis_icrs_y": axis[1],
        "spin_axis_icrs_z": axis[2],
        "spin_axis_gal_l_deg": l_gal,
        "spin_axis_gal_b_deg": b_gal,
        "L_obs_Msun_pc_kms": l_obs,
        "L_vir_Msun_pc_kms": l_vir,
        "f_L": f_l,
        "lambda_OC": lambda_oc,
        "sigma3d_kms": sig3,
        "sigma1d_kms": sig3 / math.sqrt(3.0) if np.isfinite(sig3) else np.nan,
        "strain_norm_kms_pc": float(np.linalg.norm(sym)),
        "vorticity_norm_kms_pc": float(np.linalg.norm(asym)),
    }


def bootstrap_diagnostics(
    cluster: pd.Series,
    all_pos: np.ndarray,
    masses: np.ndarray,
    rv_pos: np.ndarray,
    rv_vel: np.ndarray,
    n_boot: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    if n_boot <= 0 or len(rv_pos) < 10:
        return {}
    f_l, omega, lambdas = [], [], []
    n = len(rv_pos)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            _, grad, resid = fit_velocity_gradient(rv_pos[idx], rv_vel[idx])
            vals = diagnostic_values(cluster, all_pos, masses, grad, resid)
        except Exception:
            continue
        f_l.append(vals["f_L"])
        omega.append(vals["omega_mag_Myr_inv"])
        lambdas.append(vals["lambda_OC"])
    out = {}
    for name, values in [("f_L", f_l), ("omega_mag_Myr_inv", omega), ("lambda_OC", lambdas)]:
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        if len(arr):
            out[f"{name}_p16"] = float(np.percentile(arr, 16))
            out[f"{name}_p50"] = float(np.percentile(arr, 50))
            out[f"{name}_p84"] = float(np.percentile(arr, 84))
    return out


def null_shuffle_pvalue(
    observed_f_l: float,
    cluster: pd.Series,
    all_pos: np.ndarray,
    masses: np.ndarray,
    rv_pos: np.ndarray,
    rv_vel: np.ndarray,
    n_null: int,
    rng: np.random.Generator,
) -> float:
    if n_null <= 0 or not np.isfinite(observed_f_l):
        return np.nan
    count = 0
    valid = 0
    for _ in range(n_null):
        idx = rng.permutation(len(rv_vel))
        try:
            _, grad, resid = fit_velocity_gradient(rv_pos, rv_vel[idx])
            vals = diagnostic_values(cluster, all_pos, masses, grad, resid)
        except Exception:
            continue
        if np.isfinite(vals["f_L"]):
            valid += 1
            count += vals["f_L"] >= observed_f_l
    if valid == 0:
        return np.nan
    return float((count + 1) / (valid + 1))


def analyze_cluster(
    cluster: pd.Series,
    members: pd.DataFrame,
    min_member_prob: float,
    min_fit_rv: int,
    n_boot: int,
    n_null: int,
    use_star_parallax: bool,
    rng: np.random.Generator,
) -> dict[str, float | int | str]:
    good = finite_quality_members(members, min_member_prob)
    result: dict[str, float | int | str] = {
        "ID": int(cluster.ID),
        "Name": cluster.Name,
        "logAge50": float(cluster.logAge50),
        "age_Myr": float(10 ** (cluster.logAge50 - 6.0)),
        "N_catalog": int(cluster.N),
        "n_RV_catalog": int(cluster.n_RV),
        "MassJ_Msun": float(cluster.MassJ),
        "rJpc": float(cluster.rJpc),
        "r50Jpc": float(cluster.r50Jpc) if np.isfinite(cluster.r50Jpc) else np.nan,
        "probJ": float(cluster.probJ),
        "n_members_used": int(len(good)),
    }
    if len(good) < min_fit_rv:
        result["status"] = "too_few_quality_members"
        return result

    rv = good[
        good["RV"].notna()
        & np.isfinite(good["RV"])
        & good["e_RV"].notna()
        & (good["e_RV"] < 10)
        & ((good["RUWE"].isna()) | (good["RUWE"] < 1.4))
        & ((good["FidelityV1"].isna()) | (good["FidelityV1"] >= 0.5))
    ].copy()
    result["n_RV_fit"] = int(len(rv))
    if len(rv) < min_fit_rv:
        result["status"] = "too_few_fit_rv"
        return result

    try:
        all_pos = positions_relative_to_center(good, cluster, use_star_parallax)
        masses = good["Mass50"].to_numpy(float)
        rv_pos, rv_vel = velocities_relative_to_center(rv, cluster, use_star_parallax)
        _, gradient, residual = fit_velocity_gradient(rv_pos, rv_vel)
        result.update(diagnostic_values(cluster, all_pos, masses, gradient, residual))
        result.update(bootstrap_diagnostics(cluster, all_pos, masses, rv_pos, rv_vel, n_boot, rng))
        result["f_L_null_p"] = null_shuffle_pvalue(
            float(result["f_L"]), cluster, all_pos, masses, rv_pos, rv_vel, n_null, rng
        )
        result["status"] = "ok"
    except Exception as exc:
        result["status"] = f"failed:{type(exc).__name__}"
        result["error"] = str(exc)
    return result


def fit_decay(results: pd.DataFrame) -> dict[str, float]:
    ok = (results["status"] == "ok") & (results["f_L"] > 0) & np.isfinite(results["f_L"])
    fit = results.loc[ok].copy()
    if len(fit) < 3:
        return {"n_fit": int(len(fit)), "alpha": np.nan, "log10_A": np.nan, "scatter_dex": np.nan}
    x = np.log10(fit["age_Myr"].to_numpy(float) / 100.0)
    y = np.log10(fit["f_L"].to_numpy(float))
    coeff = np.polyfit(x, y, deg=1)
    yhat = coeff[0] * x + coeff[1]
    return {
        "n_fit": int(len(fit)),
        "alpha": float(-coeff[0]),
        "log10_A": float(coeff[1]),
        "scatter_dex": float(np.std(y - yhat, ddof=2)) if len(fit) > 2 else np.nan,
    }


def save_figures(results: pd.DataFrame, outdir: Path) -> None:
    ok = (results["status"] == "ok") & np.isfinite(results["f_L"]) & (results["f_L"] > 0)
    plot = results.loc[ok].copy()
    if plot.empty:
        return

    fit = fit_decay(plot)
    xgrid = np.logspace(np.log10(plot["age_Myr"].min() * 0.8), np.log10(plot["age_Myr"].max() * 1.2), 200)
    ygrid = 10 ** fit["log10_A"] * (xgrid / 100.0) ** (-fit["alpha"])

    plt.figure(figsize=(7, 5))
    plt.scatter(plot["age_Myr"], plot["f_L"], s=28, c=plot["n_RV_fit"], cmap="viridis", edgecolor="none")
    if np.isfinite(fit["alpha"]):
        plt.plot(xgrid, ygrid, color="black", lw=1.5, label=fr"$f_L \propto t^{{-{fit['alpha']:.2f}}}$")
        plt.legend(frameon=False)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Age (Myr)")
    plt.ylabel(r"Angular momentum retention $f_L$")
    cbar = plt.colorbar()
    cbar.set_label("RV stars in fit")
    plt.tight_layout()
    plt.savefig(outdir / "f_L_vs_age.png", dpi=220)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.scatter(plot["age_Myr"], plot["lambda_OC"], s=28, c=plot["MassJ_Msun"], cmap="plasma", edgecolor="none")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Age (Myr)")
    plt.ylabel(r"Open-cluster spin parameter $\lambda_{\rm OC}$")
    cbar = plt.colorbar()
    cbar.set_label(r"$M_J$ ($M_\odot$)")
    plt.tight_layout()
    plt.savefig(outdir / "lambda_OC_vs_age.png", dpi=220)
    plt.close()

    axes = plot[np.isfinite(plot["spin_axis_gal_l_deg"]) & np.isfinite(plot["spin_axis_gal_b_deg"])].copy()
    if not axes.empty:
        lon = np.deg2rad(((axes["spin_axis_gal_l_deg"].to_numpy(float) + 180.0) % 360.0) - 180.0)
        lat = np.deg2rad(axes["spin_axis_gal_b_deg"].to_numpy(float))
        plt.figure(figsize=(8, 4.8))
        ax = plt.subplot(111, projection="mollweide")
        sc = ax.scatter(lon, lat, c=axes["age_Myr"], s=26, cmap="cividis", edgecolor="none")
        ax.grid(True, lw=0.5)
        ax.set_xlabel("Galactic longitude")
        ax.set_ylabel("Galactic latitude")
        cbar = plt.colorbar(sc, orientation="horizontal", pad=0.08)
        cbar.set_label("Age (Myr)")
        plt.tight_layout()
        plt.savefig(outdir / "spin_axes_mollweide.png", dpi=220)
        plt.close()


def write_summary(results: pd.DataFrame, outdir: Path) -> None:
    fit = fit_decay(results)
    ok = results[results["status"] == "ok"]
    lines = [
        "# Fossil Angular Momentum Run Summary",
        "",
        f"Clusters attempted: {len(results)}",
        f"Clusters with fitted 3D gradients: {len(ok)}",
        f"Decay-fit clusters: {fit['n_fit']}",
        f"Best-fit alpha in f_L = A (t/100 Myr)^(-alpha): {fit['alpha']:.4g}",
        f"log10(A): {fit['log10_A']:.4g}",
        f"Intrinsic-looking scatter around unweighted log-log fit: {fit['scatter_dex']:.4g} dex",
        "",
        "Important caveat: this is a first-pass Gaia DR3-only measurement. It is designed",
        "to make the diagnostic reproducible, not to replace full publication-grade",
        "systematics, external RV crossmatching, or hierarchical population modelling.",
    ]
    (outdir / "run_summary.md").write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clusters", type=Path, default=Path("hunt24/clusters.dat"))
    parser.add_argument("--members", type=Path, default=Path("hunt24/members.dat"))
    parser.add_argument("--outdir", type=Path, default=Path("outputs"))
    parser.add_argument("--min-members", type=int, default=300)
    parser.add_argument("--min-catalog-rv", type=int, default=50)
    parser.add_argument("--min-fit-rv", type=int, default=30)
    parser.add_argument("--min-member-prob", type=float, default=0.5)
    parser.add_argument("--min-log-age", type=float, default=7.0)
    parser.add_argument("--max-log-age", type=float, default=9.0)
    parser.add_argument("--min-probj", type=float, default=0.5)
    parser.add_argument("--bootstrap", type=int, default=50)
    parser.add_argument("--null-trials", type=int, default=50)
    parser.add_argument(
        "--use-star-parallax",
        action="store_true",
        help="Use individual Gaia parallaxes for member distances. Default uses the cluster distance to avoid parallax-depth inflation.",
    )
    parser.add_argument("--max-clusters", type=int, default=0, help="Debug limit; 0 means all selected clusters.")
    parser.add_argument("--seed", type=int, default=20260619)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    clusters = read_fwf(args.clusters, CLUSTER_SPECS, NUMERIC_CLUSTER_COLUMNS)
    sample = select_clusters(
        clusters,
        min_members=args.min_members,
        min_catalog_rv=args.min_catalog_rv,
        min_log_age=args.min_log_age,
        max_log_age=args.max_log_age,
        min_probj=args.min_probj,
    )
    if args.max_clusters:
        sample = sample.head(args.max_clusters).copy()
    sample.to_csv(args.outdir / "sample_selection.csv", index=False)

    ids = set(sample["ID"].astype(int))
    members = read_member_chunks(args.members, MEMBER_SPECS, NUMERIC_MEMBER_COLUMNS, ids)

    rng = np.random.default_rng(args.seed)
    rows = []
    grouped = {int(k): v for k, v in members.groupby("ID")}
    for _, cluster in sample.iterrows():
        member_group = grouped.get(int(cluster.ID), pd.DataFrame(columns=members.columns))
        rows.append(
            analyze_cluster(
                cluster,
                member_group,
                min_member_prob=args.min_member_prob,
                min_fit_rv=args.min_fit_rv,
                n_boot=args.bootstrap,
                n_null=args.null_trials,
                use_star_parallax=args.use_star_parallax,
                rng=rng,
            )
        )

    results = pd.DataFrame(rows)
    results.to_csv(args.outdir / "cluster_spin_results.csv", index=False)
    save_figures(results, args.outdir)
    write_summary(results, args.outdir)

    fit = fit_decay(results)
    print(f"Selected clusters: {len(sample)}")
    print(f"Fitted clusters: {(results['status'] == 'ok').sum()}")
    print(f"Decay alpha: {fit['alpha']:.4g} from {fit['n_fit']} clusters")
    print(f"Wrote outputs to {args.outdir}")


if __name__ == "__main__":
    main()
