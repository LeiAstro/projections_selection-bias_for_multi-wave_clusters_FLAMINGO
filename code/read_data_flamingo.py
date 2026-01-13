# read_data_flamingo.py
import os
from pathlib import Path
from itertools import combinations

import numpy as np
import h5py
from astropy.io import fits

from colossus.cosmology import cosmology
from colossus.halo import profile_nfw, concentration


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


def load_halo_id(halo_prop_path):
    """
    Load halo_id array from the halo properties HDF5.
    """
    halo_prop_path = Path(halo_prop_path)
    if not halo_prop_path.exists():
        raise FileNotFoundError(f"halo_prop_path not found: {halo_prop_path}")

    with h5py.File(halo_prop_path, "r") as f:
        if "halo_id" not in f:
            raise KeyError(f"No 'halo_id' dataset in file: {halo_prop_path}")
        halo_id = f["halo_id"][:]
    return halo_id


def colossus_nfw_DS_Sigma_at_rp(
    halo_mass,           # Msun/h (IMPORTANT)
    rp_comoving,         # comoving Mpc/h
    redshift=0.3,
    mdef="500c",
    comoving=True,
    area_unit="pc2",     # "kpc2", "pc2", "Mpc2"
):
    """
    Returns (DeltaSigma(rp), Sigma(rp)) for an NFW halo using Colossus,
    evaluated at rp given in comoving Mpc/h.

    Colossus conventions:
      - profile methods take physical kpc/h radii
      - outputs are physical Msun*h/kpc^2
    """
    rp_comoving = np.asarray(rp_comoving, dtype=float)
    a = 1.0 / (1.0 + redshift)

    # comoving Mpc/h -> physical kpc/h
    r_phys_kpch = rp_comoving * a * 1e3

    c = concentration.concentration(M=halo_mass, z=redshift, mdef=mdef)
    prof = profile_nfw.NFWProfile(M=halo_mass, c=c, z=redshift, mdef=mdef)

    Sigma_phys = prof.surfaceDensity(r_phys_kpch)
    DS_phys    = prof.deltaSigma(r_phys_kpch)

    fac_a = a**2 if comoving else 1.0

    if area_unit == "kpc2":
        conv = 1.0
    elif area_unit == "pc2":
        conv = 1e-6
    elif area_unit == "Mpc2":
        conv = 1e6
    else:
        raise ValueError("area_unit must be one of {'kpc2','pc2','Mpc2'}")

    Sigma_out = Sigma_phys * fac_a * conv
    DS_out    = DS_phys    * fac_a * conv
    return DS_out, Sigma_out


def load_cylinder_cluster_richness(richness_in):
    """
    Read cylinder cluster richness FITS and return arrays sorted by haloid.
    """
    with fits.open(richness_in) as hdul:
        hdul.verify("fix")
        data_cl = hdul[1].data

    richness_cyl = data_cl["lambda"]
    cyl_m500c    = data_cl["mass_host"]
    hid_cyl      = data_cl["haloid"]

    order = np.argsort(hid_cyl)
    return richness_cyl[order], cyl_m500c[order], hid_cyl[order]


def load_cylinder_cluster_halo_matchID(matchedid_in):
    """
    Read matched cluster-halo boolean index from FITS.
    """
    with fits.open(matchedid_in) as hdul:
        hdul.verify("fix")
        data_id = hdul[1].data
    return data_id["matched_cluster_id"]


def verify_ids_match(arr_dict):
    """
    Given a dict {name: id_array}, check all pairs match exactly.
    Returns True/False (prints warnings if mismatch).
    """
    # quick length check
    lens = {k: len(v) for k, v in arr_dict.items()}
    if len(set(lens.values())) != 1:
        print("[skip] ID arrays have different lengths:", lens)
        return False

    mismatches = []
    for (name1, a1), (name2, a2) in combinations(arr_dict.items(), 2):
        if not np.array_equal(a1, a2):
            mismatches.append((name1, name2))

    if mismatches:
        for n1, n2 in mismatches:
            print(f"[skip] halo-ID mismatch between `{n1}` and `{n2}`")
        return False
    return True


def load_halo_lensing_data(data_m3_path, proj_depth, match_id, halo_id, ds_unit_factor=1e-12):
    """
    Read lensing (DeltaSigma, Sigma) and select/sort by match & halo_id.
    Returns (halo_mass_sort, DS_sort, Sigma_sort, rp_bins, halo_id_sort).
    """
    data_m3_path = str(data_m3_path)
    fn = f"lensing_DMGasStars_20rpbins_rp0.1-30.0_projdepth{proj_depth}_m500c.1e13_hpz.hdf5"
    full_path = os.path.join(data_m3_path, fn)

    with h5py.File(full_path, "r") as all_dat:
        rp_ih_all         = all_dat["r_bins"][:]          # (n_halo, n_rp) 
        DeltaSigma_ih_all = all_dat["DeltaSigma"][:] * ds_unit_factor
        Sigma_ih_all      = all_dat["Sigma"][:]      * ds_unit_factor
        halo_mass_all     = all_dat["halo_mass"][:]

    if match_id is None:
        raise ValueError("match_id must be provided!")
    if halo_id is None:
        raise ValueError("halo_id must be provided!")

    sel = (match_id == 1)
    halo_mass_matched = halo_mass_all[sel]
    DS_matched        = DeltaSigma_ih_all[sel, :]
    Sig_matched       = Sigma_ih_all[sel, :]
    hid_matched       = halo_id[sel]

    hid_order      = np.argsort(hid_matched)
    halo_mass_sort = halo_mass_matched[hid_order]
    DS_sort        = DS_matched[hid_order, :]
    Sig_sort       = Sig_matched[hid_order, :]
    hid_sort       = hid_matched[hid_order]
    rp_bins        = rp_ih_all[0, :]

    return halo_mass_sort, DS_sort, Sig_sort, rp_bins, hid_sort


def load_halo_cy_data(data_m3_path, proj_depth, match_id, halo_id):
    """
    Read Compton-y profile and select/sort by match & halo_id.
    Returns (halo_mass_sort, cy_sort, rp_bins, halo_id_sort).
    """
    data_m3_path = str(data_m3_path)
    fn = f"Compton_y_20rpbins_rp0.1-30.0_projdepth{proj_depth}_m500c.1e13_hpz.hdf5"
    full_path = os.path.join(data_m3_path, fn)

    with h5py.File(full_path, "r") as all_dat:
        rp_ih_all     = all_dat["r_bins"][:]
        cy_ih_all     = all_dat["cy_at_rp"][:]
        halo_mass_all = all_dat["halo_mass"][:]

    if match_id is None or halo_id is None:
        raise ValueError("Both match_id and halo_id must be provided!")

    sel = (match_id == 1)
    halo_mass_matched = halo_mass_all[sel]
    cy_matched        = cy_ih_all[sel, :]
    hid_matched       = halo_id[sel]

    hid_order      = np.argsort(hid_matched)
    halo_mass_sort = halo_mass_matched[hid_order]
    cy_sort        = cy_matched[hid_order, :]
    hid_sort       = hid_matched[hid_order]
    rp_bins        = rp_ih_all[0, :]

    return halo_mass_sort, cy_sort, rp_bins, hid_sort


def calc_mean_lensing_with_std_err(delta_sigma, sigma, mean_type="linear"):
    """
    Mean + standard error across halos for DeltaSigma$ and Sigma.
    mean_type: "linear" or "log"
    """
    ds = np.array(delta_sigma, dtype=float)
    sg = np.array(sigma, dtype=float)

    ds[ds <= 0] = np.nan
    sg[sg <= 0] = np.nan

    counts_ds = np.sum(~np.isnan(ds), axis=0)
    counts_sg = np.sum(~np.isnan(sg), axis=0)

    def safe_se(std, counts):
        se = np.full_like(std, np.nan)
        ok = counts > 0
        se[ok] = std[ok] / np.sqrt(counts[ok])
        return se

    if mean_type == "log":
        ln_ds   = np.log(ds)
        mean_ln = np.nanmean(ln_ds, axis=0)
        std_ln  = np.nanstd(ln_ds, axis=0, ddof=1)

        mean_DS = np.exp(mean_ln)
        err_DS  = mean_DS * safe_se(std_ln, counts_ds)

        ln_sg      = np.log(sg)
        mean_ln_sg = np.nanmean(ln_sg, axis=0)
        std_ln_sg  = np.nanstd(ln_sg, axis=0, ddof=1)

        mean_Sig = np.exp(mean_ln_sg)
        err_Sig  = mean_Sig * safe_se(std_ln_sg, counts_sg)
    else:
        mean_DS = np.nanmean(ds, axis=0)
        std_DS  = np.nanstd(ds, axis=0, ddof=1)
        err_DS  = safe_se(std_DS, counts_ds)

        mean_Sig = np.nanmean(sg, axis=0)
        std_Sig  = np.nanstd(sg, axis=0, ddof=1)
        err_Sig  = safe_se(std_Sig, counts_sg)

    return mean_DS, err_DS, mean_Sig, err_Sig


def calc_mean_cy_with_std_err(profile, mean_type="linear"):
    """
    Mean + standard error across halos for a generic positive profile (e.g., Compton-y).
    mean_type: "linear" or "log"
    """
    x = np.array(profile, dtype=float)
    x[x <= 0] = np.nan

    counts = np.sum(~np.isnan(x), axis=0)

    def safe_se(std, counts):
        se = np.full_like(std, np.nan)
        ok = counts > 0
        se[ok] = std[ok] / np.sqrt(counts[ok])
        return se

    if mean_type == "log":
        ln_x   = np.log(x)
        mean_ln = np.nanmean(ln_x, axis=0)
        std_ln  = np.nanstd(ln_x, axis=0, ddof=1)
        mean_x = np.exp(mean_ln)
        err_x  = mean_x * safe_se(std_ln, counts)
    else:
        mean_x = np.nanmean(x, axis=0)
        std_x  = np.nanstd(x, axis=0, ddof=1)
        err_x  = safe_se(std_x, counts)

    return mean_x, err_x



