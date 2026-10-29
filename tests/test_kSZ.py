"""

Test the kSZ functionality in Zeus21

Author: Claude AI
October 2026

How the file is organised
-------------------------
* One small "maps" stand-in (32^3 box, 4 redshifts) and ond KSZ object are built once (module-scoped fixtures)
  and reused by all tests. KSZ only reads: z, z_of_density, density, ncells, boxlength, ion_field_allz.
* Tests 1-5 check exact identities or self-consistency, not literature values: 
  Planck tau, SPT and AMBER need N~200-300 and several seeds, so they should be tested separately.
* A 32^3 box (L=300 cMpc) has k_Nyq ~ 0.33 Mpc^-1, so l=3000 is not resolved here: we use l <= 2000.
"""

import copy
import types

import numpy as np
import pytest
from scipy import integrate

import zeus21
from zeus21 import constants, cosmology

N, L = 32, 300.0                          # Small box for fast tests
KBOX, KNYQ = 2*np.pi/L, np.pi*N/L
Z = np.array([5.05, 7.0, 9.0, 12.0])      # Increasing redshift; Z[0] = z_of_density; NOTE: several interpolators are built over zintegral, whose first element is 5.000000000000001, and scipy’s interp1d with fill value=None returns NaN rather than extrapolating, so asking for exactly 5.0 silently yields a fully neutral box (spec Sec. 8)
ELLS = [500, 1000, 2000]
NBINS = 12
EDGES = np.logspace(np.log10(KBOX * 1.001), np.log10(KNYQ), NBINS)   # Same k shells as KSZ.__init__


# ------------------------------------------------------------------------------------------ helpers
def scalar(x):
    """Some zeus21 functions return arrays of shape (1,): turn them into a float."""
    return float(np.squeeze(x))


def kgrid(N, L):
    k1 = 2 * np.pi * np.fft.fftfreq(N, d=L / N)
    kz = 2 * np.pi * np.fft.rfftfreq(N, d=L / N)
    return k1[:, None, None], k1[None, :, None], kz[None, None, :]


def gaussian_box(cosmo, z, seed):
    """Gaussian delta with the CLASS linear power spectrum at z (same idea as powerbox in maps.generate_density)."""
    kl = cosmo._klistCF
    pk = np.array([cosmo.ClassCosmo.pk(k, z) for k in kl])              # CLASS pk is scalar-only
    kx, ky, kz = kgrid(N, L)
    k = np.sqrt(kx**2 + ky**2 + kz**2)
    k[0, 0, 0] = kl.min()
    P = np.exp(np.interp(np.log(k), np.log(kl), np.log(pk)))            # Box k-range is well inside kl
    delta_k = np.fft.rfftn(np.random.default_rng(seed).normal(size=(N, N, N))) * np.sqrt(N**3 * P / L**3)
    delta_k[0, 0, 0] = 0.0                                              # zero mean
    return np.fft.irfftn(delta_k, s=(N, N, N), axes=(0, 1, 2))


def D_ratio(cosmo, z):
    """D(z)/D(z0), computed here directly from the cosmology (not via KSZ)."""
    return scalar(cosmology.growth(cosmo, z) / cosmology.growth(cosmo, Z[0]))


def aHf(cosmo, z):
    """a H f [km/s/Mpc], computed here directly from the cosmology (not via KSZ)."""
    f = -(1.0 + z) * cosmology.dgrowth_dz(cosmo, z) / cosmology.growth(cosmo, z)
    return scalar(cosmology.Hub(cosmo, z) * f / (1.0 + z))


# ------------------------------------------------------------------------------------------ fixtures (built once)
@pytest.fixture(scope="module")
def cosmo():
    UserParams = zeus21.User_Parameters(zmin=5.0)
    return zeus21.Cosmo_Parameters(UserParams=UserParams, kmax_CLASS=100.)   # As in tests/test_maps.py


@pytest.fixture(scope="module")
def maps_small(cosmo):
    """Stand-in for reionization_maps: density at Z[0] (float32, like maps.density) and nested binary ionization fields."""
    delta = gaussian_box(cosmo, Z[0], seed=1)
    kx, ky, kz = kgrid(N, L)
    smooth = np.fft.irfftn(np.fft.rfftn(delta) * np.exp(-0.5 * (kx**2 + ky**2 + kz**2) * 6.0**2), s=(N, N, N), axes=(0, 1, 2))
    ion = np.zeros((len(Z), N, N, N))
    for iz, frac in enumerate([0.97, 0.60, 0.15, 0.0]):    # ionized volume fraction at each z; dense regions first
        if frac > 0:
            ion[iz] = smooth >= np.quantile(smooth, 1.0 - frac)
    return types.SimpleNamespace(z=Z, z_of_density=Z[0], density=delta.astype(np.float32),
                                 ncells=N, boxlength=L, ion_field_allz=ion)


@pytest.fixture(scope="module")
def ksz(cosmo, maps_small):
    return zeus21.KSZ(cosmo, maps_small, NBINS, ELLS)


# ------------------------------------------------------------------------------------------ TEST 1: Continuity
def test_continuity(ksz):
    """div(Psi) = -delta exactly (spec eq. 2), for every mode except k=0 and the Nyquist planes
    (a real field cannot satisfy it there). Expect ~1e-12 in double precision."""
    delta = np.random.default_rng(5).normal(size=(N, N, N))   # float64 input, no cosmology needed
    Psi = ksz.displacement_field(delta, N, L)
    delta_k = np.fft.rfftn(delta)
    div_k = sum(1j * ki * np.fft.rfftn(Pi) for ki, Pi in zip(ksz.kvec, Psi))
    keep = np.ones(delta_k.shape, dtype=bool)
    for ki in ksz.kvec:
        keep &= np.abs(ki) < KNYQ - 1e-9
    keep[0, 0, 0] = False
    assert (np.abs(div_k + delta_k)[keep] / np.abs(delta_k[keep])).max() < 1e-9


# ------------------------------------------------------------------------------------------ TEST 2: transverse projection
def _powers(ksz, q):
    """P_perp from KSZ.transverse_power, and the total power of q computed here (KSZ does not return it)."""
    k, P_perp = ksz.transverse_power(q, L, EDGES)
    total = sum(np.abs(np.fft.rfftn(qi))**2 for qi in q)
    _, P_tot = ksz.shell_average(total, np.sqrt(ksz.k2), EDGES, L**3/N**6)
    return P_perp, P_tot


def test_longitudinal_field_is_removed(ksz):
    """q = grad(phi) is parallel to k: its transverse power must vanish (spec: P_perp/P_tot ~ 1e-8)."""
    phi_k = np.fft.rfftn(np.random.default_rng(3).normal(size=(N, N, N)))
    q = np.array([np.fft.irfftn(1j * ki * phi_k, s=(N, N, N), axes=(0, 1, 2)) for ki in ksz.kvec])
    P_perp, P_tot = _powers(ksz, q)
    assert (P_perp / P_tot).max() < 1e-8


def test_transverse_field_is_kept(ksz):
    """Complementary check: q_k = k x a_k is perpendicular to k and must survive (guards against removing everything)."""
    a = [np.fft.rfftn(np.random.default_rng(4 + i).normal(size=(N, N, N))) for i in range(3)]
    kx, ky, kz = ksz.kvec
    q_k = [ky * a[2] - kz * a[1], kz * a[0] - kx * a[2], kx * a[1] - ky * a[0]]
    q = np.array([np.fft.irfftn(c, s=(N, N, N), axes=(0, 1, 2)) for c in q_k])
    P_perp, P_tot = _powers(ksz, q)
    assert (P_perp / P_tot).min() > 0.99


# ------------------------------------------------------------------------------------------ TEST 3: velocity amplitude and z-scaling
def test_growth_ratio(ksz, cosmo):
    """D(z)/D(z0): equal to 1 at z0, smaller than 1 above (the "silent" factor of spec Sec. 3.1)."""
    assert scalar(ksz.growth_ratio(cosmo, Z[0])) == pytest.approx(1.0, abs=1e-12)
    for z in (7.0, 12.0):
        assert scalar(ksz.growth_ratio(cosmo, z)) == pytest.approx(D_ratio(cosmo, z), rel=1e-10)
        assert scalar(ksz.growth_ratio(cosmo, z)) < 1.0


def test_velocity_scaling(ksz, cosmo, maps_small):
    """<v^2> on the box / <v^2> expected = c_v(z)^2 <Psi^2> / ([aHf D/D0]^2 int P dk/2pi^2).
    Rules from the spec: the theory side is built from the cosmology (rather than from KSZ's c_v), and only the
    z-independence of the ratio is tested (its value scatters with the realization: cosmic variance)."""
    k = np.logspace(np.log10(KBOX), np.log10(KNYQ), 2000)
    I = np.trapezoid(np.array([cosmo.ClassCosmo.pk(ki, Z[0]) for ki in k]) / (2 * np.pi**2), k)
    Psi = ksz.displacement_field(maps_small.density, N, L)
    mean_Psi2 = (Psi**2).sum(axis=0).mean()
    ratios = []
    for z in (6.0, 8.0, 10.0, 12.0):
        measured = scalar(ksz.displacement_to_velocity_conversion(cosmo, z)) ** 2 * mean_Psi2
        theory = (aHf(cosmo, z) * D_ratio(cosmo, z)) ** 2 * I
        ratios.append(measured / theory)
    assert np.ptp(ratios) / np.mean(ratios) < 1e-2          # spec: < 1% drift with z
    assert 0.2 < np.mean(ratios) < 5.0                      # Loose sanity check only


def test_velocity_conversion_at_z0(ksz, cosmo):
    """At z0 the growth ratio is 1, so c_v(z0) = a H f exactly [km/s per Mpc]."""
    assert scalar(ksz.displacement_to_velocity_conversion(cosmo, Z[0])) == pytest.approx(aHf(cosmo, Z[0]), rel=1e-10)


# ------------------------------------------------------------------------------------------ TEST 4: ionization field
def test_xe_binary_reads_the_right_field(ksz, maps_small):
    for iz in range(len(Z)):
        assert np.array_equal(ksz.xe_binary(maps_small, iz), maps_small.ion_field_allz[iz] > 0.5)


def test_xbar_mass_definition(ksz, maps_small, cosmo):
    """Mass-weighted history: <x_e (1+delta_L)> / <1+delta_L>, delta_L = D/D0 delta_box, 1+delta_L clipped at 0 (spec Sec. 4)."""
    xbar = ksz.xbar_mass
    for iz, z in enumerate(Z):
        dens = np.clip(1.0 + D_ratio(cosmo, z) * maps_small.density.astype(np.float64), 0.0, None)
        xe = maps_small.ion_field_allz[iz] > 0.5
        assert xbar[iz] == pytest.approx((xe * dens).mean() / dens.mean(), rel=1e-6, abs=1e-12)  # rel 1e-6: float32 density


# ------------------------------------------------------------------------------------------ TEST 5: missing large-scale modes (Park+13)
def test_v2_missing(ksz, cosmo):
    """<v^2>_miss = (aHf)^2 int_0^kbox P(k,z) dk/2pi^2 (spec eq. 7): converged in Npoints, and equal to an
    independent adaptive integral (including the analytic tail below kmin)."""
    z = 8.0
    assert ksz.v2_missing(cosmo, z, KBOX) == pytest.approx(ksz.v2_missing(cosmo, z, KBOX, Npoints=4000), rel=1e-3)
    kmin = 2e-2 * KBOX                                               # = kmin_tol * k_box (default kmin_tol)
    pk = lambda kk: cosmo.ClassCosmo.pk(kk, z)
    I, _ = integrate.quad(lambda lk: np.exp(lk) * pk(np.exp(lk)) / (2 * np.pi**2), np.log(kmin), np.log(KBOX))
    I += pk(kmin) * kmin / (1.0 + cosmo.ns) / (2 * np.pi**2)         # Power-law tail below kmin
    expected = aHf(cosmo, z) ** 2 * I / constants.c_kms**2
    assert scalar(ksz.v2_missing(cosmo, z, KBOX, extend_below_kmin=True)) == pytest.approx(expected, rel=3e-3)


def test_park_term_in_the_pipeline(cosmo, maps_small, ksz):
    """The Park term stored by the pipeline must be (2/3) <v^2>_miss P_chi(k), with P_chi recomputed here (spec test 5)."""
    iz, z = 1, Z[1]
    dens = np.clip(1.0 + D_ratio(cosmo, z) * maps_small.density.astype(np.float64), 0.0, None)
    chi = (maps_small.ion_field_allz[iz] > 0.5) * dens
    _, P_chi = ksz.shell_average(np.abs(np.fft.rfftn(chi)) ** 2, np.sqrt(ksz.k2), EDGES, L**3 / N**6)
    expected = (2.0 / 3.0) * ksz.v2_missing(cosmo, z, KBOX) * P_chi
    assert ksz.P_k_park[iz] == pytest.approx(expected, rel=1e-5)


# ------------------------------------------------------------------------------------------ TEST 6 (mechanics only): optical depth
def test_tau_integral(ksz, cosmo):
    """tau_of_z (spec eq. 10) for x_e = 1 up to z=15 vs an independent quadrature, split at the He II -> He III step (z=3).
    The comparison with Planck is a validation, not a unit test."""
    zgrid = np.linspace(5.05, 15.0, 400)
    tau = ksz.tau_of_z(cosmo, 15.0, zgrid, np.ones_like(zgrid))
    pref = constants.sigmaT * constants.Mpctocm * constants.c_kms * cosmology.n_H(cosmo, 0.0)
    f = lambda z: (1 + z) ** 2 * (1 + cosmo.x_He + cosmo.x_He * (z < 3.0)) / cosmology.Hub(cosmo, z)
    expected = pref * (integrate.quad(f, 0.0, 3.0)[0] + integrate.quad(f, 3.0, 15.0)[0])
    assert tau == pytest.approx(expected, rel=1e-3)


# ------------------------------------------------------------------------------------------ Limber integral (spec eq. 8)
def test_limber_against_direct_integral(ksz, cosmo):
    """Synthetic P_q_perp = const and tau = 0: checks the 1/2, a^-4, ds = c dz/H, units, ell(ell+1)/2pi and T_CMB^2."""
    ell, P0 = 1000, 3.0e-12
    zgrid = np.linspace(5.05, 15.0, 400)
    k_list = [np.logspace(-3, 1, 50)] * len(zgrid)
    P_list = [np.full(50, P0)] * len(zgrid)
    D, contrib = ksz.limber(cosmo, zgrid, [ell], k_list, P_list, np.zeros_like(zgrid))
    pref = (constants.sigmaT * constants.Mpctocm * cosmology.n_H(cosmo, 0.0) * (1 + cosmo.x_He)) ** 2
    dC = lambda z: pref * constants.c_kms / cosmology.Hub(cosmo, z) * (1 + z) ** 4 / scalar(cosmology.chi_of_redshift(cosmo, z)) ** 2 * P0 / 2
    expected = ell * (ell + 1) * cosmo.ClassCosmo.T_cmb() ** 2 / (2 * np.pi) * 1e12 * integrate.quad(dC, 5.05, 15.0)[0]
    assert D[0] == pytest.approx(expected, rel=1e-3)
    assert np.trapezoid(contrib[0], zgrid) == pytest.approx(D[0], rel=1e-12)    # D_ell = integral of its z-contributions


def test_loglog_interp():
    k = np.logspace(-2, 0, 20)
    assert zeus21.KSZ.loglog_interp(0.1, k, k**-2.5) == pytest.approx(0.1**-2.5, rel=1e-12)   # Exact for a power law
    with pytest.raises(ValueError):
        zeus21.KSZ.loglog_interp(2.0, k, k**-2.5)                     # Refuses to extrapolate outside the box band
    assert zeus21.KSZ.loglog_interp(0.1, k, np.zeros_like(k)) == 0.0  # No ionized cells -> no contribution


# ------------------------------------------------------------------------------------------ Whole object: sanity
def test_outputs_are_sane(ksz):
    assert np.all(np.isfinite(ksz.D_ell)) and np.all(ksz.D_ell > 0)
    assert np.all(np.diff(ksz.tau) >= 0)                             # tau grows with z


def test_park_term_adds_power(cosmo, maps_small, ksz):
    assert np.all(ksz.D_ell > zeus21.KSZ(cosmo, maps_small, NBINS, ELLS, include_Park_correction=False).D_ell)


def test_rejects_decreasing_redshifts(cosmo, maps_small):
    bad = copy.copy(maps_small)
    bad.z = Z[::-1]
    with pytest.raises(AssertionError):
        zeus21.KSZ(cosmo, bad, NBINS, ELLS)
