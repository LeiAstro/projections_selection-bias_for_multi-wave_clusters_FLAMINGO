# expected_profiles_wu2022.py
import os
# from pathlib import Path
# from itertools import combinations

import numpy as np
# import h5py
# from astropy.io import fits

from colossus.cosmology import cosmology
# from colossus.halo import profile_nfw, concentration


def setup_flamingo_cosmology():
    """
    Set Flamingo cosmology in Colossus.
    """
    flamingo_params = {
        "flat": True,
        "H0": 68.1,
        "Om0": 0.306,
        "Ob0": 0.0486,
        "sigma8": 0.807,
        "ns": 0.967,
    }
    # Colossus will raise if name exists; handle gracefully
    try:
        cosmology.addCosmology("flamingo", flamingo_params)
    except Exception:
        pass
    return cosmology.setCosmology("flamingo")


# def ln_mass_binning(mass, dlnM=0.05, edges=None):
#     """
#     Return lnM, bin edges in lnM, bin centers in M, and bin index per halo.
#     """
#     mass = np.asarray(mass, float)
#     lnM = np.log(mass)

#     if edges is None:
#         edges = np.arange(lnM.min(), lnM.max() + dlnM, dlnM)
#         if edges[-1] < lnM.max():
#             edges = np.append(edges, lnM.max() + 1e-12)
#     edges = np.asarray(edges, float)

#     nbin = len(edges) - 1
#     bin_idx = np.digitize(lnM, edges, right=False) - 1
#     bin_idx = np.clip(bin_idx, 0, nbin - 1)

#     # mass centers (not log centers): exp of midpoints in ln-space
#     M_centers = np.exp(0.5 * (edges[:-1] + edges[1:]))

#     return lnM, edges, M_centers, bin_idx

############################################################################
############################################################################
# ------------------------- mass-bin weighting ----------------------------

def mass_pdf_weighted_lensing_with_sampling_err(
    mass,
    delta_sigma,
    rich_mask,
    dlogM=0.05,
    mean_type="linear",   # "linear" or "log"
    edges=None            # Option: pass existing mass edges array for exact matching
):
    """
    Wu+2022 Appendix B method (iii):
    Expected profile from *mass-PDF only* (old version):
      - compute mean profile in each mass bin using ALL halos
      - weight those bin-means by the mass-PDF of richness-selected halos
      - estimate sampling error from per-bin variance and counts

    Returns
    -------
    ds_mean, ds_err, M_centers, pdf_rich
    """
    
    mass      = np.asarray(mass, float)
    ds_all    = np.asarray(delta_sigma, float)
    rich_mask = np.asarray(rich_mask, bool)

    lnM = np.log(mass)

    # define edges in ln(M)
    if edges is None:
        edges = np.arange(lnM.min(), lnM.max() + dlogM, dlogM)
        if edges[-1] <= lnM.max():
            edges = np.append(edges, lnM.max() + 1e-12)
    edges = np.asarray(edges, float)

    n_bins = len(edges) - 1
    M_centers = np.exp(0.5 * (edges[:-1] + edges[1:]))  # exp(midpoints in ln-space)
    n_rad = ds_all.shape[1]

    # bin index
    bin_idx = np.digitize(lnM, edges, right=False) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    # selected-sample mass PDF (this is the obs rich-sel mass "blue histogram" PDF)
    counts_sel = np.bincount(bin_idx[rich_mask], minlength=n_bins)
    tot = counts_sel.sum()
    if tot == 0:
        raise ValueError("No halos pass rich_mask!")
    pdf_sel = counts_sel / float(tot)

    val_mean = np.zeros((n_bins, n_rad), float)
    var_i    = np.zeros_like(val_mean)
    N_i      = np.zeros(n_bins, int)

    for i in range(n_bins):
        sel = (bin_idx == i)          # ALL halos in bin i
        N_i[i] = sel.sum()
        if N_i[i] == 0:
            continue

        x = ds_all[sel]

        if mean_type == "log":
            x = np.where(x > 0, x, np.nan)
            x = np.log(x)

        val_mean[i] = np.nanmean(x, axis=0)

        if N_i[i] > 1:
            var_i[i] = np.nanvar(x, axis=0, ddof=1)
        else:
            var_i[i].fill(0.0)

    # expected mean across bins
    if mean_type == "log":
        mean_log = np.average(val_mean, axis=0, weights=pdf_sel)
        ds_mean  = np.exp(mean_log)
    else:
        ds_mean  = np.average(val_mean, axis=0, weights=pdf_sel)

    # sampling variance (Wu+22 method iii style)
    w2_over_N = np.where(N_i > 0, (pdf_sel**2) / N_i, 0.0)
    var_sampling = np.sum(w2_over_N[:, None] * var_i, axis=0)

    if mean_type == "log":
        ds_err = ds_mean * np.sqrt(var_sampling)
    else:
        ds_err = np.sqrt(var_sampling)

    return ds_mean, ds_err, M_centers, pdf_sel


# ------------------------- Per-halo weighting ----------------------------
def mass_pdf_weighted_per_halo_lensing(
    mass,
    profile,
    rich_mask,
    dlogM=0.05,
    mean_type="linear",     # "linear" or "log"
    edges=None,             # Option: pass existing mass edges array for exact matching
    return_weights=False
):
    """
    Wu+2022 Appendix B method (iii):
    Compute an 'expected' stacked profile using ALL halos, but weight each halo
    by the observed mass-PDF of the richness-selected sample (blue histogram)---New version.

    This is algebraically equivalent to:
        expected = sum_bins pdf_obs(bin) * mean_all_halos_in_bin(profile)

    Parameters
    ----------
    mass : (N,) array
        Halo masses (e.g., M500c) for all halos (same ordering as profile rows).
    profile : (N, R) array
        Per-halo radial profile (e.g., DeltaSigma, Sigma, cy).
    rich_mask : (N,) bool array
        True for halos in the observed richness-selected sample (used to build pdf_obs).
    dlogM : float
        Width of log/ln mass bins.
    mean_type : {"linear", "log"}
        "linear": weighted arithmetic mean in linear space.
        "log"   : weighted mean of log(profile), then exponentiate.
    edges : array or None
        Bin edges in log/ln(M). If None, computed from full mass range using dlogM.
        Pass the same edges that are used elsewhere to ensure exact reproducibility.
    return_weights : bool
        If True, also return per-halo weights.

    Returns
    -------
    prof_exp : (R,) array
        Expected stacked profile.
    prof_err : (R,) array
        Sampling error estimate (same form as the 'mass_pdf_weighted_lensing_with_sampling_err' bin-variance propagation).
    edges : (nbin+1,) array
        ln(M) bin edges used.
    pdf_obs : (nbin,) array
        Observed mass PDF of the richness-selected sample.
    (optional) w_halo : (N,) array
        Per-halo weights used in the stack.
    """
    mass = np.asarray(mass)
    prof = np.asarray(profile, dtype=float)
    rich_mask = np.asarray(rich_mask, dtype=bool)

    if prof.ndim != 2:
        raise ValueError("profile must be a 2D array with shape (N_halo, N_rp).")
    if mass.shape[0] != prof.shape[0] or mass.shape[0] != rich_mask.shape[0]:
        raise ValueError("mass, profile, rich_mask must have the same first dimension.")

    logM = np.log(mass) ### Note: in python, log=ln

    # Define bin edges in log(M)
    if edges is None:
        edges = np.arange(logM.min(), logM.max() + dlogM, dlogM)
        if edges[-1] <= logM.max():
            edges = np.append(edges, logM.max() + 1e-12)
    edges = np.asarray(edges, dtype=float)
    nbin = len(edges) - 1

    # Bin index for each halo: 0..nbin-1
    # right=False means [edges[i], edges[i+1]) except last edge handling; clip to be safe
    bin_idx = np.digitize(logM, edges, right=False) - 1
    bin_idx = np.clip(bin_idx, 0, nbin - 1)

    # Observed PDF from richness-selected sample
    counts_sel = np.bincount(bin_idx[rich_mask], minlength=nbin)
    tot_sel = counts_sel.sum()
    if tot_sel == 0:
        raise ValueError("No halos pass rich_mask; cannot form observed PDF.")
    pdf_obs = counts_sel / float(tot_sel)

    # All-halo counts per bin (for normalization of per-halo weights)
    counts_all = np.bincount(bin_idx, minlength=nbin)
    w_bin = np.zeros(nbin, dtype=float)
    ok = counts_all > 0
    w_bin[ok] = pdf_obs[ok] / counts_all[ok]   # per-halo weight within each mass bin
    w_halo = w_bin[bin_idx]                    # (N,)

    # Weighted mean profile across all halos
    if mean_type == "log":
        x = prof.copy()
        x[x <= 0] = np.nan
        x = np.log(x)
    else:
        x = prof

    # compute weighted mean per radius with NaN-handling
    prof_exp = np.full(prof.shape[1], np.nan, dtype=float)
    for r in range(prof.shape[1]):
        xr = x[:, r]
        m = np.isfinite(xr) & np.isfinite(w_halo) & (w_halo > 0)
        if not np.any(m):
            continue
        wr = w_halo[m]
        prof_exp[r] = np.sum(wr * xr[m]) / np.sum(wr)

    if mean_type == "log":
        prof_exp = np.exp(prof_exp)

    # ---- Sampling error: match the previous bin-variance propagation ----
    # var_sampling = sum_i (pdf_obs^2 / N_all_i) * var_i
    # where var_i is per-bin variance of x (linear or log) across ALL halos in that bin.
    R = prof.shape[1]
    var_i = np.zeros((nbin, R), dtype=float)

    for i in range(nbin):
        sel_i = (bin_idx == i)
        Ni = sel_i.sum()
        if Ni <= 1:
            continue
        xi = x[sel_i, :]
        var_i[i, :] = np.nanvar(xi, axis=0, ddof=1)

    w2_over_N = np.zeros(nbin, dtype=float)
    w2_over_N[ok] = (pdf_obs[ok] ** 2) / counts_all[ok]
    var_sampling = np.sum(w2_over_N[:, None] * var_i, axis=0)

    if mean_type == "log":
        prof_err = prof_exp * np.sqrt(var_sampling)  # var_sampling is in log-space
    else:
        prof_err = np.sqrt(var_sampling)

    if return_weights:
        return prof_exp, prof_err, edges, pdf_obs, w_halo
    return prof_exp, prof_err, edges, pdf_obs

############################################################################
############################################################################

def _stack_from_samples(samples_2d, mean_type="linear"):
    x = np.array(samples_2d, dtype=float)
    x[x <= 0] = np.nan

    counts = np.sum(~np.isnan(x), axis=0)

    def safe_se(std, counts):
        se = np.full_like(std, np.nan)
        ok = counts > 0
        se[ok] = std[ok] / np.sqrt(counts[ok])
        return se

    if mean_type == "log":
        lx = np.log(x)
        mean_l = np.nanmean(lx, axis=0)
        std_l  = np.nanstd(lx, axis=0, ddof=1)
        mean   = np.exp(mean_l)
        err    = mean * safe_se(std_l, counts)
    else:
        mean = np.nanmean(x, axis=0)
        std  = np.nanstd(x, axis=0, ddof=1)
        err  = safe_se(std, counts)
    return mean, err

############################################################################

def mass_matched_sample_expected(
    mass,
    profile,
    sel_mask,
    dlogM=0.05,
    factor=5, ### Default is same as Wu+2022
    n_real=50,
    mean_type="linear",
    edges=None,
    seed=0,
):
    """
    Wu+2022 Appendix B method (ii): build a random 'mass-matched' sample
    that matches the mass PDF of the richness-selected sample (sel_mask),
    with total size = factor * N_selected (draw with replacement if needed). 
    *** Note: NOT exclude the observed halos ***
    Returns
    -------
    exp_mean : (N_rp,) mean over realizations of the mass-matched stack
    exp_std  : (N_rp,) std over realizations (MC noise of the matching)
    exp_all  : (n_real, N_rp) stacked profiles for each realization
    """
    mass = np.asarray(mass, float)
    prof = np.asarray(profile, float)
    sel_mask = np.asarray(sel_mask, bool)
    rng = np.random.default_rng(seed)

    lnM = np.log(mass)
    if edges is None:
        edges = np.arange(lnM.min(), lnM.max() + dlogM, dlogM)
        if edges[-1] <= lnM.max():
            edges = np.append(edges, lnM.max() + 1e-12)
    edges = np.asarray(edges, float)
    nbin = len(edges) - 1

    b = np.digitize(lnM, edges, right=False) - 1
    b = np.clip(b, 0, nbin - 1)

    counts_sel = np.bincount(b[sel_mask], minlength=nbin)
    if counts_sel.sum() == 0:
        raise ValueError("sel_mask selects zero halos.")
    # counts_tgt = factor * counts_sel
    counts_tgt = np.rint(factor * counts_sel).astype(int)

    idx_all = np.arange(len(mass))
    bin_to_idx = [idx_all[b == i] for i in range(nbin)]

    exp_all = np.zeros((n_real, prof.shape[1]), float)

    for t in range(n_real):
        draw_idx = []
        for i in range(nbin):
            k = int(counts_tgt[i])
            if k <= 0:
                continue
            pool = bin_to_idx[i]
            if pool.size == 0:
                raise ValueError(f"Mass bin {i} has 0 halos but needs {k} draws.")
            replace = (k > pool.size)
            draw_idx.append(rng.choice(pool, size=k, replace=replace))
        # in case, the sample is empty    
        if len(draw_idx) == 0:
            exp_all[t, :] = np.nan
            continue

        draw_idx = np.concatenate(draw_idx)

        exp_all[t], _ = _stack_from_samples(prof[draw_idx], mean_type=mean_type)

    exp_mean = np.nanmean(exp_all, axis=0)
    exp_std  = np.nanstd(exp_all, axis=0, ddof=1)
    return exp_mean, exp_std, exp_all

############################################################################
############################################################################

def stack_profile_mean(profile_2d, mask, mean_type="linear"):
    x = np.array(profile_2d, dtype=float)[mask]

    # match the function calc_mean_lensing_with_std_err
    # drop <=0 always
    x[x <= 0] = np.nan

    counts = np.sum(~np.isnan(x), axis=0)

    def safe_se(std, counts):
        se = np.full_like(std, np.nan)
        ok = counts > 0
        se[ok] = std[ok] / np.sqrt(counts[ok])
        return se

    if mean_type == "log":
        lx = np.log(x)
        mean_l = np.nanmean(lx, axis=0)
        std_l  = np.nanstd(lx, axis=0, ddof=1)
        mean   = np.exp(mean_l)
        err    = mean * safe_se(std_l, counts)
    else:
        mean = np.nanmean(x, axis=0)
        std  = np.nanstd(x, axis=0, ddof=1)
        err  = safe_se(std, counts)

    return mean, err

############################################################################
def shuffled_richness_selection_bias_ratio(
    mass,
    richness,
    profile,
    lam_range=None,      # (lam_lo, lam_hi) apply same thresholds to lambda and lambda_shuff
    sel_mask=None,       # OR provide sel_mask and use rank-slice selection below
    rank_slice=None,     # (start_rank, end_rank) on sorted richness (same counts) — matches the DESY1-counts style
    obs_mask=None,
    dlogM=0.05,
    n_shuffle=100,
    mean_type="linear",
    edges=None,
    seed=0,
):
    """
    Wu+2022 Appendix B method (i): shuffle richness within narrow mass bins
    to erase correlated residuals between richness and lensing at fixed mass. :contentReference[oaicite:3]{index=3}

    Then compute ratio:
        R(rp) = <lensing(profile) | selected by lambda> / <lensing(profile) | selected by lambda_shuff>

    Selection options (pick ONE):
      A) lam_range=(lam_lo, lam_hi): select by richness thresholds (faithful to "richness bin")
      B) sel_mask: if provided, uses same-N selection by rank (requires rank_slice=None; see below)
      C) rank_slice=(start,end): select by richness rank interval (fits the fixed-count bins)

    Returns
    -------
    ratio_mean : (N_rp,) mean over shuffles
    ratio_std  : (N_rp,) std over shuffles
    """
    mass = np.asarray(mass, float)
    lam  = np.asarray(richness, float)
    prof = np.asarray(profile, float)
    rng  = np.random.default_rng(seed)

    logM = np.log(mass)
    if edges is None:
        edges = np.arange(logM.min(), logM.max() + dlogM, dlogM)
        if edges[-1] <= logM.max():
            edges = np.append(edges, logM.max() + 1e-12)
    edges = np.asarray(edges, float)
    nbin = len(edges) - 1

    b = np.digitize(logM, edges, right=False) - 1
    b = np.clip(b, 0, nbin - 1)

    def make_mask_from_lambda(lam_vals):
        if lam_range is not None:
            lo, hi = lam_range
            return (lam_vals >= lo) & (lam_vals < hi)
        if rank_slice is not None:
            s, e = rank_slice
            order = np.argsort(lam_vals)[::-1]  # descending richness
            mask = np.zeros_like(lam_vals, dtype=bool)
            mask[order[s:e]] = True
            return mask
        if sel_mask is not None:
            # use same N as sel_mask by selecting top-N in lam_vals
            N = int(np.sum(sel_mask))
            if N <= 0:
                raise ValueError("sel_mask selects zero halos.")
            order = np.argsort(lam_vals)[::-1]
            mask = np.zeros_like(lam_vals, dtype=bool)
            mask[order[:N]] = True
            return mask
        raise ValueError("Provide lam_range or rank_slice or sel_mask.")

    # observed selection mask and observed stack
    # if obs_mask is not None:
    #     m_obs = np.asarray(obs_mask, dtype=bool)
    # else:
    #     m_obs = make_mask_from_lambda(lam)

    if obs_mask is not None:
        m_obs = np.asarray(obs_mask, dtype=bool)
        if m_obs.shape[0] != mass.shape[0]:
            raise ValueError("obs_mask must have same length as mass.")
        if m_obs.sum() == 0:
            raise ValueError("obs_mask selects zero halos.")
    else:
        m_obs = make_mask_from_lambda(lam)

    prof_obs, _ = stack_profile_mean(prof, m_obs, mean_type=mean_type)


    # shuffle richness within each mass bin
    ratio_all = np.zeros((n_shuffle, prof.shape[1]), float)
    idx_all = np.arange(len(mass))

    for t in range(n_shuffle):
        lam_shuff = lam.copy()
        for i in range(nbin):
            idx = idx_all[b == i]
            if idx.size >= 2:
                lam_shuff[idx] = rng.permutation(lam_shuff[idx])

        m_sh = make_mask_from_lambda(lam_shuff)
        prof_sh, _ = stack_profile_mean(prof, m_sh, mean_type=mean_type)

        # ratio_all[t] = prof_obs / prof_sh
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio_all[t] = prof_obs / prof_sh

    ratio_mean = np.nanmean(ratio_all, axis=0)
    ratio_std  = np.nanstd(ratio_all, axis=0, ddof=1)
    return ratio_mean, ratio_std

