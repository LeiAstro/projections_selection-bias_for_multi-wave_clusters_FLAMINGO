import numpy as np
import healpy as hp
import astropy.units as u
from astropy.table import Table
# from calc_volume_numdens_scaled_richness import make_astropy_flamingo_cosmo

from utils.calc_volume_numdens_scaled_richness import (
    make_astropy_flamingo_cosmo,
)


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


def volume_weighted_mean_area_DESY1(
    calc,
    z1,
    z2,
    *,
    n_eval=256,
    cosmo=None,
    return_volume=False,
    return_details=False,
):
    """
    Volume-weighted mean area <A> over [z1, z2], where A(z) is provided by `calc`.

    Returns:
      - default: A_volw_deg2 (float)
      - if return_volume=True: dict with A_volw_deg2, V_eff_total, V_shell_per_deg2
      - if return_details=True: also includes z_eval, A_z_deg2, dV_dz_deg2
    """
    if cosmo is None:
        cosmo = make_astropy_flamingo_cosmo()

    z1 = float(z1)
    z2 = float(z2)
    if z2 <= z1:
        raise ValueError("Require z2 > z1")

    h = cosmo.H0.value / 100.0
    sr_per_deg2 = (np.pi / 180.0) ** 2

    zz = np.linspace(z1, z2, int(n_eval))

    # A(z) in deg^2
    A_deg2 = calc.area_deg2_at_many_z(zz)

    # dV/dz per deg^2 in (Mpc/h)^3
    dV_dz_sr = cosmo.differential_comoving_volume(zz).to_value(u.Mpc**3 / u.sr)
    dV_dz_deg2 = dV_dz_sr * sr_per_deg2 * h**3  # (Mpc/h)^3 / deg^2 / dz

    V_eff_total = float(np.trapz(A_deg2 * dV_dz_deg2, zz))     # (Mpc/h)^3
    V_shell_per_deg2 = float(np.trapz(dV_dz_deg2, zz))         # (Mpc/h)^3 per deg^2

    A_volw = 0.0 if V_shell_per_deg2 == 0.0 else V_eff_total / V_shell_per_deg2

    if not return_volume and not return_details:
        return A_volw

    out = {
        "A_volw_deg2": A_volw,
        "V_eff_total": V_eff_total,
        "V_shell_per_deg2": V_shell_per_deg2,
    }
    if return_details:
        out.update({
            "z_eval": zz,
            "A_z_deg2": A_deg2,
            "dV_dz_deg2": dV_dz_deg2,
        })
    return out

def effective_volume_bins_DESY1(calc, z_edges, *, n_eval=256, cosmo=None):
    """
    For each z bin [z_edges[i], z_edges[i+1]], compute:
      V_eff_bins[i] = ∫ A(z) dV/dz_deg2 dz   (Mpc/h)^3
    Also returns:
      V_shell_per_deg2_bins[i] = ∫ dV/dz_deg2 dz  (Mpc/h)^3 per deg^2
      A_volw_bins[i] = V_eff_bins / V_shell_per_deg2_bins  (deg^2)
      V_eff_total, A_volw_total

    Example call it:
    z_edges = np.linspace(0.2, 0.65, 10)
    out = effective_volume_bins_DESY1(calc, z_edges)
    
    print(out["V_eff_total"])
    print(out["V_eff_bins"])

    """
    if cosmo is None:
        cosmo = make_astropy_flamingo_cosmo()

    z_edges = np.asarray(z_edges, float)
    if np.any(~np.isfinite(z_edges)) or np.any(np.diff(z_edges) <= 0):
        raise ValueError("z_edges must be finite and strictly increasing")

    nb = len(z_edges) - 1
    V_eff_bins = np.zeros(nb, float)
    V_shell_per_deg2_bins = np.zeros(nb, float)
    A_volw_bins = np.zeros(nb, float)

    for i in range(nb):
        z1, z2 = z_edges[i], z_edges[i+1]
        r = volume_weighted_mean_area_DESY1(
            calc, z1, z2, n_eval=n_eval, cosmo=cosmo, return_volume=True
        )
        V_eff_bins[i] = r["V_eff_total"]
        V_shell_per_deg2_bins[i] = r["V_shell_per_deg2"]
        A_volw_bins[i] = r["A_volw_deg2"]

    V_eff_total = float(np.sum(V_eff_bins))
    V_shell_per_deg2_total = float(np.sum(V_shell_per_deg2_bins))
    A_volw_total = 0.0 if V_shell_per_deg2_total == 0.0 else V_eff_total / V_shell_per_deg2_total

    return {
        "z_edges": z_edges,
        "V_eff_bins": V_eff_bins,
        "V_shell_per_deg2_bins": V_shell_per_deg2_bins,
        "A_volw_bins_deg2": A_volw_bins,
        "V_eff_total": V_eff_total,
        "A_volw_total_deg2": A_volw_total,
    }

# def volume_weighted_mean_area_DESY1(calc, z1, z2, *, n_eval=256, cosmo=None):
#     """
#     Volume-weighted mean area <A> over [z1, z2], where A(z) is provided by `calc`.
#     Build `calc` on the (possibly filtered) (zmax, fracgood) arrays you want to use.
#     """
#     if cosmo is None:
#         cosmo = make_astropy_flamingo_cosmo()

#     h = cosmo.H0.value / 100.0
#     sr_per_deg2 = (np.pi / 180.0) ** 2

#     zz = np.linspace(z1, z2, int(n_eval))

#     # A(z) in deg^2 (fast)
#     A_deg2 = calc.area_deg2_at_many_z(zz)

#     # dV/dz per deg^2 in (Mpc/h)^3
#     dV_dz_sr = cosmo.differential_comoving_volume(zz).to_value(u.Mpc**3 / u.sr)
#     dV_dz_deg2 = dV_dz_sr * sr_per_deg2 * h**3

#     num = np.trapz(A_deg2 * dV_dz_deg2, zz)
#     den = np.trapz(dV_dz_deg2, zz)
#     return num / den
