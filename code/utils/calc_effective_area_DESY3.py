import numpy as np
import healpy as hp
import astropy.units as u
from astropy.table import Table
# from calc_volume_numdens_scaled_richness import make_astropy_flamingo_cosmo

from utils.calc_volume_numdens_scaled_richness import (
    make_astropy_flamingo_cosmo,
)
########################
###### For DES Y3 ######
########################

def solve_M_from_NK(N, K, *, max_iter=80, tol=1e-10):
    """
    Goal: correct for empty pixels
    Solve E[K] = M * (1 - (1 - 1/M)^N) for M, given N points and K occupied pixels.
    Estimate the underlying number of HEALPix pixels M given:
      N = number of points thrown uniformly into M pixels
      K = number of distinct (occupied) pixels observed
    Using: E[K] = M * (1 - (1 - 1/M)^N)
    Solve for M by bisection. Requires 1 <= K <= min(N, M).
    Bisection; stable for large N, M.
    """
    N = int(N); K = int(K)
    if K <= 0 or N <= 0:
        return 0.0
    if K == 1:
        return 1.0

    def EK(M):
        M = float(M)
        # (1 - 1/M)^N in stable form
        return M * (1.0 - np.exp(N * np.log1p(-1.0 / M)))

    lo = float(K)
    hi = max(float(K) * 2.0, float(K) + 1.0)
    while EK(hi) < K:
        hi *= 2.0
        if hi > 1e16:
            break

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        val = EK(mid)
        if abs(val - K) / max(K, 1) < tol:
            return mid
        if val < K:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)

def _pix_from_radec(ra_deg, dec_deg, nside, nest=False):
    theta = np.deg2rad(90.0 - np.asarray(dec_deg))
    phi   = np.deg2rad(np.asarray(ra_deg) % 360.0)
    return hp.ang2pix(nside, theta, phi, nest=nest)

def area_vs_z_bins_DES_weighted(
    rand_ra, rand_dec, rand_ztrue, rand_weight,
    z_edges,
    *,
    nside=512,
    nest=False,
    base_mask=None,         # e.g., random_select mask
    select_mask=None,       # e.g., (avg_lambdaout in [20,30))
    min_points=1000,
    show_empty_fraction=False,
    
):
    """
    Returns per redshift bin:
      A_geom_deg2  : occupancy-corrected geometric area (deg^2)
      f_surv       : N / sum(w)  (kept/trials)  [DES/redMaPPer consistent]
      A_eff_deg2   : A_geom * f_surv
    If select_mask provided:
      f_sel        : N_sel / sum(w)
      A_eff_sel    : A_geom * f_sel
    """
    ra = np.asarray(rand_ra)
    dec = np.asarray(rand_dec)
    z = np.asarray(rand_ztrue)
    w = np.asarray(rand_weight)
    z_edges = np.asarray(z_edges, dtype=float)

    if base_mask is None:
        base_mask = np.ones_like(z, dtype=bool)
    else:
        base_mask = np.asarray(base_mask, dtype=bool)

    if select_mask is not None:
        select_mask = np.asarray(select_mask, dtype=bool)
        if select_mask.shape != z.shape:
            raise ValueError("select_mask must have same shape as inputs.")

    nb = len(z_edges) - 1
    zc = 0.5 * (z_edges[:-1] + z_edges[1:])

    A_pix = hp.nside2pixarea(nside, degrees=True)

    A_geom = np.zeros(nb, float)
    f_surv = np.full(nb, np.nan, float)
    A_eff  = np.zeros(nb, float)

    f_sel = None
    A_eff_sel = None
    if select_mask is not None:
        f_sel = np.full(nb, np.nan, float)
        A_eff_sel = np.zeros(nb, float)

    dbg = []

    for i in range(nb):
        zlo, zhi = z_edges[i], z_edges[i+1]
        m = base_mask & (z >= zlo) & (z < zhi)
        N = int(np.sum(m))
        if N < min_points:
            dbg.append({"bin": i, "zlo": zlo, "zhi": zhi, "N": N, "note": "too_few_points"})
            continue

        pix = _pix_from_radec(ra[m], dec[m], nside=nside, nest=nest)
        K = np.unique(pix).size

        # occupancy correction (empty pixels)
        Mhat = solve_M_from_NK(N, K)
        occ = N / Mhat
        if show_empty_fraction is True:
            print("N/Mhat =", occ, "empty fraction ~", np.exp(-occ),f"for redshift bin: [{zlo}, {zhi})")

        A_geom[i] = Mhat * A_pix

        # DES/redMaPPer survival fraction (kept / trials)
        Wsum = float(np.sum(w[m]))
        f_surv[i] = N / Wsum
        A_eff[i] = A_geom[i] * f_surv[i]

        info = {"bin": i, "zlo": float(zlo), "zhi": float(zhi), "N": N, "K": int(K), "Mhat": float(Mhat),
                "A_pix": float(A_pix), "Wsum": Wsum, "f_surv": float(f_surv[i])}

        if select_mask is not None:
            ms = m & select_mask
            Nsel = int(np.sum(ms))
            f_sel[i] = Nsel / Wsum
            A_eff_sel[i] = A_geom[i] * f_sel[i]
            info.update({"Nsel": Nsel, "f_sel": float(f_sel[i])})

        dbg.append(info)

    out = {"z_center": zc, "A_geom_deg2": A_geom, "f_surv": f_surv, "A_eff_deg2": A_eff, "debug": dbg}
    if select_mask is not None:
        out["f_sel"] = f_sel
        out["A_eff_deg2"] = A_eff_sel
    return out


# import numpy as np

# def y3_volume_weighted_area_and_volume(z_edges, Aeff_deg2, *, per_deg2_shell_volume_func):
#     """
#     z_edges: array-like of bin edges
#     Aeff_deg2: array-like A_eff in deg^2 per bin (same length as bins)
#     per_deg2_shell_volume_func(zlo, zhi) -> (Mpc/h)^3 per deg^2 in that shell

#     Returns:
#       A_volw_deg2: volume-weighted mean effective area
#       V_eff_total: total effective volume in (Mpc/h)^3 across the z range
#       V_shell_per_deg2: array of per-deg^2 shell volumes
#       V_eff_bins: array of effective volumes per bin
#     """
#     z_edges = np.asarray(z_edges, float)
#     Aeff = np.asarray(Aeff_deg2, float)
#     nb = len(z_edges) - 1
#     assert len(Aeff) == nb

#     V_shell_per_deg2 = np.array([per_deg2_shell_volume_func(z_edges[i], z_edges[i+1]) for i in range(nb)])
#     V_eff_bins = Aeff * V_shell_per_deg2
#     V_eff_total = np.sum(V_eff_bins)

#     A_volw = np.sum(Aeff * V_shell_per_deg2) / np.sum(V_shell_per_deg2)
#     return A_volw, V_eff_total, V_shell_per_deg2, V_eff_bins


# import numpy as np
# import astropy.units as u

def shell_volume_per_deg2_astropy(z_edges, cosmo):
    """
    Per-bin comoving volume per deg^2 in (Mpc/h)^3.
    z_edges: array of bin edges.
    cosmo: astropy cosmology (e.g., make_astropy_flamingo_cosmo()).
    """
    z_edges = np.asarray(z_edges, float)
    # comoving volume (Mpc^3) enclosed within z, full-sky:
    V_fullsky = cosmo.comoving_volume(z_edges).to_value(u.Mpc**3)  # shape (nb+1,)
    # shell volume full-sky:
    dV_fullsky = V_fullsky[1:] - V_fullsky[:-1]                   # Mpc^3

    # convert to per-deg^2:
    area_sr_per_deg2 = (np.pi/180.0)**2
    V_per_deg2_Mpc3 = dV_fullsky * (area_sr_per_deg2 / (4.0*np.pi))

    # convert Mpc^3 -> (Mpc/h)^3
    h = cosmo.H0.value / 100.0
    V_per_deg2 = V_per_deg2_Mpc3 * h**3
    return V_per_deg2  # (Mpc/h)^3 per deg^2 in each bin


def volume_weighted_area_and_volume(z_edges, A_eff_deg2, cosmo, *, ignore_nan=True):
    """
    Combine A_eff(z-bin) with shell volume weights to produce:
      V_eff_bins  (Mpc/h)^3 per bin
      V_eff_total (Mpc/h)^3 total
      A_volw      (deg^2) volume-weighted mean area
    """
    z_edges = np.asarray(z_edges, float)
    A_eff = np.asarray(A_eff_deg2, float)
    nb = len(z_edges) - 1
    if A_eff.shape[0] != nb:
        raise ValueError("A_eff_deg2 must have length len(z_edges)-1")

    V_shell_per_deg2 = shell_volume_per_deg2_astropy(z_edges, cosmo)  # (Mpc/h)^3 per deg^2
    V_eff_bins = A_eff * V_shell_per_deg2                              # (Mpc/h)^3

    if ignore_nan:
        m = np.isfinite(V_eff_bins) & np.isfinite(V_shell_per_deg2) & np.isfinite(A_eff)
        V_eff_total = np.sum(V_eff_bins[m])
        A_volw = np.sum(A_eff[m] * V_shell_per_deg2[m]) / np.sum(V_shell_per_deg2[m])
    else:
        V_eff_total = np.sum(V_eff_bins)
        A_volw = np.sum(A_eff * V_shell_per_deg2) / np.sum(V_shell_per_deg2)

    return {
        "V_shell_per_deg2": V_shell_per_deg2,
        "V_eff_bins": V_eff_bins,
        "V_eff_total": V_eff_total,
        "A_volw_deg2": A_volw,
    }


########################
###### For DES Y1 ######
########################

def load_y1_redmapper_zmask(zmask_fits_gz, *, hpx_col="HPIX", zmax_col="ZMAX", frac_col="FRACGOOD"):
    """
    Load the Y1 redMaPPer zmask (footprint) file.
    Returns (hpix, zmax, fracgood) as numpy arrays.
    """
    tab = Table.read(zmask_fits_gz)  # works with .fits or .fits.gz
    hpix = np.asarray(tab[hpx_col])
    zmax = np.asarray(tab[zmax_col], dtype=float)
    frac = np.asarray(tab[frac_col], dtype=float)

    # Clean / guard
    m = np.isfinite(zmax) & np.isfinite(frac) & (frac > 0)
    return hpix[m], zmax[m], frac[m]

class ZMaskAreaCalculator:
    """
    **Only for DES Y1 footprint mask file**
    Fast A(z) queries using sorting + cumulative sums.
    A(z) = A_pix * sum(fracgood[zmax>=z]).
    """
    def __init__(self, zmax, fracgood, *, nside=4096):
        self.nside = int(nside)
        self.A_pix = float(hp.nside2pixarea(self.nside, degrees=True))

        order = np.argsort(zmax)              # ascending zmax
        self.zmax_sorted = np.asarray(zmax)[order]
        frac_sorted = np.asarray(fracgood)[order]
        self.cum_frac = np.cumsum(frac_sorted)
        self.total_frac = float(self.cum_frac[-1]) if self.cum_frac.size else 0.0

    def area_deg2_at_z(self, z):
        """
        Area (deg^2) available for detection at redshift z.
        """
        z = float(z)
        if self.total_frac == 0.0:
            return 0.0
        idx = np.searchsorted(self.zmax_sorted, z, side="left")
        frac_below = float(self.cum_frac[idx-1]) if idx > 0 else 0.0
        frac_keep = self.total_frac - frac_below
        return frac_keep * self.A_pix

    def area_deg2_at_many_z(self, z_array):
        z_array = np.asarray(z_array, float)
        if self.total_frac == 0.0:
            return np.zeros_like(z_array)

        idx = np.searchsorted(self.zmax_sorted, z_array, side="left")
        # frac_below = cum_frac[idx-1] with idx==0 -> 0
        frac_below = np.zeros_like(z_array, float)
        m = idx > 0
        frac_below[m] = self.cum_frac[idx[m]-1]
        frac_keep = self.total_frac - frac_below
        return frac_keep * self.A_pix


def volume_weighted_mean_area_DESY1(calc, z1, z2, zmax, *, n_eval=256, fracgood_min=0.8, binary=False):
    # calc must provide zmax_sorted/fracgood info OR just pass zmax/fracgood directly.
    # Specific for DES Y1 file, which provides arrays zmax, fracgood, hpix.
    cosmo = make_astropy_flamingo_cosmo()
    h = cosmo.H0.value / 100.0
    sr_per_deg2 = (np.pi/180.0)**2

    zz = np.linspace(z1, z2, n_eval)

    # A(z) from zmask
    A_pix = hp.nside2pixarea(4096, degrees=True)
    A = np.zeros_like(zz)

    for i, z in enumerate(zz):
        m = (zmax >= z) & (fracgood >= fracgood_min)
        if binary:
            A[i] = A_pix * np.sum(m)
        else:
            A[i] = A_pix * np.sum(fracgood[m])

    # dV/dz per deg^2 in (Mpc/h)^3
    dV_dz_sr = cosmo.differential_comoving_volume(zz).to_value(u.Mpc**3/u.sr)
    dV_dz_deg2 = dV_dz_sr * sr_per_deg2 * h**3

    num = np.trapz(A * dV_dz_deg2, zz)
    den = np.trapz(dV_dz_deg2, zz)
    return num / den
