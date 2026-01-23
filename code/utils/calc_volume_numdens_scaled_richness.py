import numpy as np
import astropy.units as u
from astropy.cosmology import FlatLambdaCDM
# from colossus.cosmology import cosmology
import treecorr
import logging
try:
    from colossus.cosmology import cosmology as ccosmo
    HAS_COLOSSUS = True
except ImportError:
    HAS_COLOSSUS = False


# -----------------------------
# Flamingo cosmology parameters
# -----------------------------
FLAMINGO = dict(
    H0=68.1,      # km/s/Mpc
    Om0=0.306,
    Ob0=0.0486,
    sigma8=0.807,
    ns=0.967,
    Tcmb0=2.7255, # K
)

def flamingo_h():
    return FLAMINGO["H0"] / 100.0

def make_astropy_flamingo_cosmo():
    return FlatLambdaCDM(
        H0=FLAMINGO["H0"] * u.km / u.s / u.Mpc,
        Om0=FLAMINGO["Om0"],
        Tcmb0=FLAMINGO["Tcmb0"] * u.K,
    )


def f_sky_from_area_deg2(sky_area_deg2: float) -> float:
    """
    Convert survey area in deg^2 to sky fraction f_sky.
    Uses astropy.units if available; otherwise uses the exact conversion.
    """
    try:
        import astropy.units as u
        area_sr = (sky_area_deg2 * u.deg**2).to(u.sr).value
    except ImportError:
        # 1 deg = pi/180 rad => 1 deg^2 = (pi/180)^2 sr
        area_sr = sky_area_deg2 * (np.pi / 180.0) ** 2

    return area_sr / (4.0 * np.pi)

# ---------------------------------------------
# Method 1: "Simple" (Astropy comoving_volume)
# ---------------------------------------------
def volume_simple_astropy_comoving_volume(z_min, z_max, area_deg2, *, verbose=True):
    """
    Uses astropy comoving_volume (integrated exactly inside astropy).
    Returns V_survey in (Mpc/h)^3 and R_equiv in (Mpc/h).
    """
    cosmo = make_astropy_flamingo_cosmo()
    h = cosmo.H0.value / 100.0
    f_sky = f_sky_from_area_deg2(area_deg2)

    V_shell = (cosmo.comoving_volume(z_max) - cosmo.comoving_volume(z_min)) * f_sky  # Mpc^3
    V_Mpc3 = V_shell.to_value(u.Mpc**3)

    V = V_Mpc3 * h**3        # (Mpc/h)^3
    R = V ** (1.0 / 3.0)     # (Mpc/h)

    if verbose:
        print(f"[Simple/Astropy comoving_volume] V = {V:.9e} (Mpc/h)^3, R = {R:.6f} (Mpc/h)")

    return V, R


# ---------------------------------------------------
# Method 2: Astropy distance + analytic shell volume
# ---------------------------------------------------
def volume_astropy_distance_shell(z_min, z_max, area_deg2, *, verbose=True):
    """
    Computes r(z) from astropy comoving_distance, then uses V = 4/3*pi*(r_max^3-r_min^3).
    Returns V_survey in (Mpc/h)^3 and R_equiv in (Mpc/h).
    """
    cosmo = make_astropy_flamingo_cosmo()
    h = cosmo.H0.value / 100.0
    f_sky = f_sky_from_area_deg2(area_deg2)

    r_min_Mpc = cosmo.comoving_distance(z_min).to_value(u.Mpc)
    r_max_Mpc = cosmo.comoving_distance(z_max).to_value(u.Mpc)

    V_allsky_Mpc3 = (4.0 / 3.0) * np.pi * (r_max_Mpc**3 - r_min_Mpc**3)
    V_survey_Mpc3 = V_allsky_Mpc3 * f_sky

    V = V_survey_Mpc3 * h**3
    R = V ** (1.0 / 3.0)

    if verbose:
        print("-" * 80)
        print(f"[Astropy distance+shell] z={z_min:.3f}: chi={r_min_Mpc*h:.6f} (Mpc/h), {r_min_Mpc:.6f} (Mpc)")
        print(f"[Astropy distance+shell] z={z_max:.3f}: chi={r_max_Mpc*h:.6f} (Mpc/h), {r_max_Mpc:.6f} (Mpc)")
        print(f"[Astropy distance+shell] V = {V:.9e} (Mpc/h)^3, R = {R:.6f} (Mpc/h)")

    return V, R


# ----------------------------------------------------
# Method 3: Colossus distance + analytic shell volume
# ----------------------------------------------------
def get_colossus_flamingo_cosmo():
    params = {
        "flat": True,
        "H0": FLAMINGO["H0"],
        "Om0": FLAMINGO["Om0"],
        "Ob0": FLAMINGO["Ob0"],
        "sigma8": FLAMINGO["sigma8"],
        "ns": FLAMINGO["ns"],
    }
    # Guard for re-running in Jupyter notebooks
    try:
        ccosmo.addCosmology("flamingo", params)
    except Exception:
        pass
    return ccosmo.setCosmology("flamingo")


def volume_colossus_distance_shell(z_min, z_max, area_deg2, *, verbose=True):
    """
    Colossus comovingDistance returns Mpc/h directly.
    Returns V_survey in (Mpc/h)^3 and R_equiv in (Mpc/h).
    """
    if not HAS_COLOSSUS:
        raise ImportError("Colossus is not installed/importable in this environment.")

    cosmo = get_colossus_flamingo_cosmo()
    h = cosmo.h
    f_sky = f_sky_from_area_deg2(area_deg2)

    r_min = cosmo.comovingDistance(0.0, z_min)  # Mpc/h
    r_max = cosmo.comovingDistance(0.0, z_max)  # Mpc/h

    V_allsky = (4.0 / 3.0) * np.pi * (r_max**3 - r_min**3)  # (Mpc/h)^3
    V = V_allsky * f_sky
    R = V ** (1.0 / 3.0)

    if verbose:
        print("-" * 80)
        print(f"[Colossus distance+shell] z={z_min:.3f}: chi={r_min:.6f} (Mpc/h), {r_min/h:.6f} (Mpc)")
        print(f"[Colossus distance+shell] z={z_max:.3f}: chi={r_max:.6f} (Mpc/h), {r_max/h:.6f} (Mpc)")
        print(f"[Colossus distance+shell] V = {V:.9e} (Mpc/h)^3, R = {R:.6f} (Mpc/h)")

    return V, R


#############################################################

def assign_jk_labels(ra_list, dec_list, npatches):
    """
    Generate spatial Jackknife labels using treecorr KMeans clustering.
    """
    # allow passing single 1-D arrays:
    if isinstance(ra_list, np.ndarray):
        ra_list = [ra_list]
        dec_list = [dec_list]

    if len(ra_list) != len(dec_list):
        logging.error("ra_list and dec_list must have the same length.")
        return None

    all_ra = np.concatenate(ra_list)
    all_dec = np.concatenate(dec_list)

    # Check for matching lengths of concatenated arrays
    if len(all_ra) != len(all_dec):
        logging.error("Concatenated RA and Dec arrays must have the same length.")
        return None

    cat = treecorr.Catalog(ra=all_ra, dec=all_dec, ra_units='deg', dec_units='deg')
    field = cat.getNField()
    logging.info("Running KMeans clustering for Jackknife labels...")
    all_labels = field.run_kmeans(npatches)[0]

    labels_list = []
    start = 0
    for ra in ra_list:
        end = start + len(ra)
        labels = all_labels[start:end]
        labels_list.append(labels)
        start = end

    return labels_list