import numpy as np
from utils.read_data_flamingo import setup_flamingo_cosmology, calc_mean_cy_with_std_err
from scipy.special import j1


import os
import h5py
from datetime import datetime

cosmo = setup_flamingo_cosmology()
# h = cosmo.H0 / 100.0
# print(h)

# ---------- Check rp/theta bins ----------
rp_min, rp_max, n_bins = 0.1, 30.0, 20  # Mpc/h
rp_edges = np.logspace(np.log10(rp_min), np.log10(rp_max), n_bins+1)
# rp_mid_test = np.sqrt(rp_edges[1:]*rp_edges[:-1]) 
# print(rp_edges)
# print(rp_mid_test)

# rpmid_list = np.sqrt(R_comv_bins_inner[:-1] * R_comv_bins_inner[1:])[1:] #Unit: cMpc/h
# print(rp_ih_all[0,:])


chi = cosmo.comovingDistance(0.0, 0.3)
theta_edges = rp_edges / chi  # radians
#--------------------------------------------

def load_Nyy_cmbs4(filepath):
    """CMB-S4 file: columns [ell] [N_TT] [N_yy]. Return ell, Nyy."""
    dat = np.loadtxt(filepath)
    ell = dat[:, 0]
    Nyy = dat[:, 2]
    return ell, Nyy

def load_Nyy_so(filepath, deproj=0):
    """
    SO file: columns [ell] [Deproj-0] [Deproj-1] [Deproj-2] [Deproj-3]
    I used Deproj-0, and this is the fiducial one.
    Return ell, Nyy for chosen deproj.
    """
    dat = np.loadtxt(filepath)
    ell = dat[:, 0]
    if deproj not in (0, 1, 2, 3):
        raise ValueError("deproj must be 0,1,2,3")
    Nyy = dat[:, 1 + deproj]
    return ell, Nyy

def extend_low_ell(ell, Nyy, ell_min_target=2, mode="flat", alpha=2.0):
    ### low ell often noise
    ell = np.asarray(ell)
    Nyy = np.asarray(Nyy)

    ell_min_file = int(np.min(ell))
    if mode == "none" or ell_min_file <= ell_min_target:
        return ell, Nyy

    ell_lo = np.arange(ell_min_target, ell_min_file)
    if mode == "flat":
        Nyy_lo = np.full_like(ell_lo, Nyy[0], dtype=float)
    elif mode == "power":
        Nyy_lo = Nyy[0] * (ell_min_file / ell_lo) ** alpha
    else:
        raise ValueError("mode must be 'none', 'flat', or 'power'")

    return np.concatenate([ell_lo, ell]), np.concatenate([Nyy_lo, Nyy])


def W_annulus(ell, th_in, th_out):
    denom = (th_out**2 - th_in**2)
    W = np.zeros_like(ell, dtype=float)
    m = ell > 0
    W[m] = 2.0 * (th_out * j1(ell[m] * th_out) - th_in * j1(ell[m] * th_in)) / (ell[m] * denom)
    W[~m] = 1.0
    return W


def cov_profile_from_Nyy(ell, Nyy, theta_edges):
    ell = np.asarray(ell)
    Nyy = np.asarray(Nyy)
    nb = len(theta_edges) - 1

    W = np.vstack([W_annulus(ell, theta_edges[i], theta_edges[i+1]) for i in range(nb)])
    pref = (ell / (2.0 * np.pi)) * Nyy

    C = np.zeros((nb, nb), dtype=float)
    for i in range(nb):
        for j in range(i, nb):
            Cij = np.trapz(pref * W[i] * W[j], ell)
            C[i, j] = C[j, i] = Cij

    # Symmetrize (important for numerical stability)
    C = 0.5 * (C + C.T)
    return C


def make_pd_cholesky(C, jitter0=0.0, jitter_rel=1e-12, max_tries=12):
    """
    Return (L, jitter_used) where L is Cholesky factor of C + jitter*I.
    We try increasing jitter until Cholesky succeeds.
    """
    C = 0.5 * (C + C.T)  # enforce symmetry
    diag_scale = float(np.mean(np.diag(C))) if np.all(np.isfinite(np.diag(C))) else 1.0
    diag_scale = max(diag_scale, 1e-30)

    # Start jitter: either provided jitter0 or a relative tiny fraction of diag_scale
    jitter = max(jitter0, jitter_rel * diag_scale)

    for k in range(max_tries):
        try:
            L = np.linalg.cholesky(C + jitter * np.eye(C.shape[0]))
            return L, jitter
        except np.linalg.LinAlgError:
            jitter *= 10.0  # increase aggressively

    # Last resort: eigenvalue clipping to PSD then chol
    w, V = np.linalg.eigh(C)
    w_clipped = np.clip(w, 1e-30, None)
    C_psd = (V * w_clipped) @ V.T
    C_psd = 0.5 * (C_psd + C_psd.T)
    # With PSD enforced, a tiny jitter should work
    jitter = max(jitter, 1e-12 * diag_scale)
    L = np.linalg.cholesky(C_psd + jitter * np.eye(C.shape[0]))
    return L, jitter


class YProfileNoiseAdder:
    """
    Build profile covariance from Nyy(ell), then add correlated Gaussian noise.
    Robust to near-singular / not-PD numeric issues.
    """
    def __init__(
        self,
        theta_edges,
        ell,
        Nyy,
        seed=426789,
        ell_extend_mode="flat",
        ell_min_target=2,
        alpha=2.0,
        jitter0=0.0,
        jitter_rel=1e-12,
        max_tries=12,
    ):
        self.theta_edges = np.asarray(theta_edges)

        ell2, Nyy2 = extend_low_ell(
            ell, Nyy, ell_min_target=ell_min_target, mode=ell_extend_mode, alpha=alpha
        )
        self.ell = np.asarray(ell2)
        self.Nyy = np.asarray(Nyy2)

        self.C = cov_profile_from_Nyy(self.ell, self.Nyy, self.theta_edges)

        self.L, self.jitter_used = make_pd_cholesky(
            self.C, jitter0=jitter0, jitter_rel=jitter_rel, max_tries=max_tries
        )

        self.rng = np.random.default_rng(seed)

    def add_to_profile(self, y_profile):
        y_profile = np.asarray(y_profile)
        nb = self.C.shape[0]
        if y_profile.shape[0] != nb:
            raise ValueError(f"y_profile length {y_profile.shape[0]} != nbins {nb}")
        z = self.rng.standard_normal(nb)
        return y_profile + (self.L @ z)

    def add_to_many(self, Y_profiles):
        Y = np.asarray(Y_profiles)
        nb = self.C.shape[0]
        if Y.ndim != 2 or Y.shape[1] != nb:
            raise ValueError(f"Y_profiles must have shape (N, {nb}), got {Y.shape}")
        Z = self.rng.standard_normal((Y.shape[0], nb))
        noise = Z @ self.L.T
        return Y + noise


######################################################################################
### Save the noise Compton-y profiles ###

def save_NOISEcomptony_profiles_hdf5(
    savepath,
    rp_edges=None,                 # (nbin+1,) optional
    rp_centers=None,               # (nbin,) optional
    rp_per_halo=None,              # (Nhalo, nbin) optional
    cy_s4=None,                    # (Nhalo, nbin) optional
    cy_so_baseline=None,           # (Nhalo, nbin) optional
    cy_so_goal=None,               # (Nhalo, nbin) optional
    halo_mass=None,                # (Nhalo,) optional
    meta=None,                     # dict of attributes
    overwrite=True,
    compression="gzip",
    compression_opts=4,
    shuffle=True,
):
    """
    Save Compton-y profiles (optionally with multiple noise models) to HDF5.

    - Uses HDF5 'w' mode to overwrite by default.
    - Adds compression + chunking for large arrays.
    - Stores metadata as file attributes.
    """

    mode = "w" if overwrite else "w-"

    def _dset_kwargs(arr):
        # chunk along halo dimension for fast per-halo reads
        arr = np.asarray(arr)
        if arr.ndim == 2:
            chunks = (min(arr.shape[0], 2048), arr.shape[1])
        elif arr.ndim == 1:
            chunks = (min(arr.shape[0], 8192),)
        else:
            chunks = True
        return dict(
            compression=compression,
            compression_opts=compression_opts,
            shuffle=shuffle,
            chunks=chunks,
        )

    with h5py.File(savepath, mode) as f:
        # ---- metadata ----
        f.attrs["created_utc"] = datetime.utcnow().isoformat()
        if meta is not None:
            for k, v in meta.items():
                # h5py attrs like scalars/strings; convert arrays to string if needed
                try:
                    f.attrs[k] = v
                except TypeError:
                    f.attrs[k] = str(v)

        # ---- radii ----
        if rp_edges is not None:
            f.create_dataset("rp_edges_comv_mpc_h", data=np.asarray(rp_edges), **_dset_kwargs(rp_edges))
        if rp_centers is not None:
            f.create_dataset("rp_centers_comv_mpc_h", data=np.asarray(rp_centers), **_dset_kwargs(rp_centers))
        if rp_per_halo is not None:
            f.create_dataset("rp_comv_mpc_h", data=np.asarray(rp_per_halo), **_dset_kwargs(rp_per_halo))

        # ---- profiles ----
        if cy_s4 is not None:
            f.create_dataset("cy_s4_noise", data=np.asarray(cy_s4), **_dset_kwargs(cy_s4))
        if cy_so_baseline is not None:
            f.create_dataset("cy_so_baseline_noise", data=np.asarray(cy_so_baseline), **_dset_kwargs(cy_so_baseline))
        if cy_so_goal is not None:
            f.create_dataset("cy_so_goal_noise", data=np.asarray(cy_so_goal), **_dset_kwargs(cy_so_goal))

        # ---- halo properties ----
        if halo_mass is not None:
            f.create_dataset("halo_mass_msun_h", data=np.asarray(halo_mass), **_dset_kwargs(halo_mass))

    print(f"[INFO] Saved Compton-y profiles to: {savepath}")

