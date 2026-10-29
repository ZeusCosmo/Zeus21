"""

Kinetic Sunyaev-Zeldovich calculation.

Authors: zeus21 v2 collaboration - October 2026
    Alessandra Venditti ;
    Julian B. Muñoz
Venditti et al. (in prep)
"""

from . import constants
from . import cosmology
from . import maps

import numpy as np

class KSZ:

    """
    kSZ power spectrum.
    
    Parameters
    ----------
    UserParams : UserParams class
    CosmoParams : CosmoParams class
    
    Attributes
    ----------
    z: 1D np.array
        Redshifts at which the outputs are computed
    xbar_mass : 1D np.array
        Mass-weighted ionization history of the box at the redshift of interest
    tau : 1D np.array
        Values of the integrated optical depth at the redshifts of interest
    k : 2D np.array
        Power spectrum wavenumbers
    P_k_resolved : 2D np.array
        Resolved transverse power spectrum at each k (computed directly from the box)
    P_k_park : 2D np.array
        Large-scale modes of the power spectrum beyond the box size, from the Park+13 correction
    ell : 1D np.array
        l values to compute the kSZ power spectrum amplitude at the redshifts of interest
    D_ell : 2D np.array
        Amplitudes of the kSZ power spectrum for each specified l value at the redshifts of interest
    D_ell_per_z : 2D np.array
        Integrand of D_ell, i.e. contribution per unit z, used for the cumulative D_ell(>z)
    """

    def __init__(self, CosmoParams, maps, nbins, ells, include_Park_correction=True):

        self.ell = np.array(ells)
        # TODO: discuss if better to build the maps object here (mirroring the T21 coefficient calculation), or externally
        self.z = maps.z   # TODO: or z_of_density? Should we allow the user to input? Single z or array?
        self.z0 = maps.z_of_density
        delta_box = maps.density
        N_box = maps.ncells
        L_box = maps.boxlength

        assert np.all(np.diff(self.z) > 0), "maps.z must be strictly increasing (z[0] = z_of_density = lowest z)"
        k_box, k_Nyq = 2*np.pi/L_box,  np.pi*N_box/L_box   # TODO: what are these? And edges?
        edges = np.logspace(np.log10(k_box * 1.001), np.log10(k_Nyq), nbins)

        self.kvec, self.k2 = self.k_grid(N_box, L_box)   # TODO: check if can do using maps.compute_k(), right now that function only computes the k2 term... + better to compute once here, or to recompute in the displacement field and transverse momentum functions? This avoids recalculations, but potential inconsistency if the delta and q on the provided grid don't match...
        
        displacement = self.displacement_field(delta_box, N_box, L_box)   # Displacement field from Zeldovich approximation, computed once, at z0 --> TODO: check actual redshift of maps.density

        k_list, P_k_resolved, P_k_park, xbar_mass = [], [], [], []
        for iz, z in enumerate(self.z):
            xe = self.xe_binary(maps, iz)                                                         # Binary ionization field at this z
            delta_L = self.growth_ratio(CosmoParams, z) * delta_box                               # Linear density at z: growth enters here
            dens = np.clip(1.0 + delta_L, 0.0, None)                                              # delta_L clipped to avoid negative gas density
            chi = xe * dens                                                                       # chi = x_e (1 + delta_L): electron density / mean --> TODO: check if can use maps.ion_frac, if mass/volume weighted and how to introduce partials
            v = displacement * self.displacement_to_velocity_conversion(CosmoParams, z)           # Velocity field
            q = chi[None] * v/constants.c_kms                                                     # Momentum field, scheme E (Eulerian density) --> NOTE: chi has shape (N,N,N), v (3,N,N,N); chi[None] ensures correct alignment, avoiding silent errors if incorrect data layout
            k, P_q_perp = self.transverse_power(q, L_box, edges)                                  # Resolved part of the spectrum
            k_list.append(k); P_k_resolved.append(P_q_perp)
            if include_Park_correction:                                                           # Missing-mode term (Park+13)
                _, P_chi = self.scalar_power(chi, L_box, edges)
                dP_park = (2./3.)*self.v2_missing(CosmoParams, z, k_box)*P_chi
                P_k_park.append(dP_park)
            else:
                P_k_park.append(np.zeros_like(P_q_perp))                                                            
            xbar_mass.append(chi.mean()/dens.mean())                                              # mass-weighted <(1+delta) x_e> / <1+delta>
            #xbar_vol.append(xe.mean())                                                           # volume-weighted (just to show the difference)
        self.xbar_mass = np.array(xbar_mass)
        self.k = np.array(k_list)
        self.P_k_resolved = np.array(P_k_resolved)
        self.P_k_park = np.array(P_k_park)

        self.tau = self.tau_of_z(CosmoParams, self.z, self.z, self.xbar_mass)                          # tau(z) from the box's own history (same x_e as in q)
        self.D_ell, self.D_ell_per_z = self.limber(CosmoParams, self.z, self.ell, self.k, self.P_k_resolved + self.P_k_park, self.tau)


    @staticmethod
    def k_grid(N, L):
        """Wavenumber components for an rfftn-transformed N^3 box of side L [Mpc].
        Returns ((kx, ky, kz), k2) with shapes (N,1,1), (1,N,1), (1,1,N//2+1) --> they BROADCAST, no meshgrid.
        kz only has the non-negative half because rfftn stores half of Fourier space (the other half is the complex conjugate).
        """
        k1  = 2 * np.pi * np.fft.fftfreq(N, d=L/N)   # Full range, for the two full axes
        kz1 = 2 * np.pi * np.fft.rfftfreq(N, d=L/N)   # Non-negative half, for the last axis
        kx, ky, kz = k1[:, None, None], k1[None, :, None], kz1[None, None, :]
        return (kx, ky, kz), kx**2 + ky**2 + kz**2

    def displacement_field(self, delta, N, L):
        """Zel'dovich displacement Psi [cMpc] from the density box: Psi_k = i k delta_k / k^2 (spec eq. 2).
        Returns an array of shape (3, N, N, N) (components x, y, z). Psi is built at the same redshift where delta is defined.
        """
        delta_k = np.fft.rfftn(delta)
        Psi = []
        k2_safe = self.k2.copy(); k2_safe[0, 0, 0] = 1.0   # Avoid 0/0 at k=0
        for ki in self.kvec:   # One component at a time
            Psi_k = 1j * ki * delta_k / k2_safe
            Psi_k[0, 0, 0] = 0.0   # Set Psi(k=0) = 0 explicitly
            Psi.append(np.fft.irfftn(Psi_k, s=(N, N, N), axes=(0, 1, 2)))
        return np.array(Psi)


    @staticmethod
    def xe_binary(maps, iz):
        """Binary ionization field at the iz-th redshift of maps.z (spec eq. 4)."""
        return np.asarray(maps.ion_field_allz[iz]) > 0.5   # TODO: fix maps.compute_zreion_frombinaryxHII() calculation: this assigns z[-1] (the HIGHEST z) to cells that never ionize, instead of NaN


    def growth_ratio(self, CosmoParams, z):
        """Linear growth factor normalised to D(0)=1. Only RATIOS D(z)/D(z0) matter below."""
        return cosmology.growth(CosmoParams, z) / cosmology.growth(CosmoParams, self.z0)   # TODO: for this and growth_ratio_logderivative, check consistency with maps.py, as in that case CosmoParams.growthint is used, which does not contain the correction for the 21cmFAST emulation

    @staticmethod
    def growth_ratio_logderivative(CosmoParams, z):
        """Growth rate f = dlnD/dlna = -(1+z) dlnD/dz (spec eq. 1), by central finite differences.
        During reionization f ~ 1."""
        f = -(1.0 + z) * cosmology.dgrowth_dz(CosmoParams, z) / cosmology.growth(CosmoParams, z)
        return f[0] if np.isscalar(z) else f

    def displacement_to_velocity_conversion(self, CosmoParams, z):
        """Velocity conversion factor c_v(z) = a H f D(z)/D(z0) [km/s per Mpc of displacement] (spec eq. 3).
        """
        a = 1.0 / (1.0 + z)
        return a * cosmology.Hub(CosmoParams, z) * self.growth_ratio_logderivative(CosmoParams, z) * self.growth_ratio(CosmoParams, z)


    @staticmethod
    def shell_average(field_k, kmag, edges, norm):
        """Average a Fourier-space quantity in |k| shells and multiply by norm (= L^3/N^6).
        Returns (mean |k| in each non-empty shell, power). Uses bincount: fast and memory friendly.
        All elements of the rfft half-space carry equal weight; this is unbiased because every mode in a shell has the same expectation.
        """
        idx = np.digitize(kmag, edges) - 1
        ok = (idx >= 0) & (idx < len(edges) - 1)   # drops k=0 and modes beyond the last edge
        count = np.bincount(idx[ok], minlength=len(edges) - 1)
        power = np.bincount(idx[ok], weights=field_k[ok], minlength=len(edges) - 1) / np.maximum(count, 1)
        keff  = np.bincount(idx[ok], weights=kmag[ok],    minlength=len(edges) - 1) / np.maximum(count, 1)
        good = count > 0
        return keff[good], norm*power[good]

    def transverse_power(self, q, L, edges):
        """Transverse power spectrum P_q⊥(k) of a real vector field q with shape (3,N,N,N) (spec eq. 6).
        Returns k, P_perp(k), P_total(k). P_total = power of q itself, used only for testing purposes (test 2).
        """
        N = q.shape[-1]
        q_k = [np.fft.rfftn(q[i]) for i in range(3)]                  # rfftn: q is real
        k_dot_q = sum(ki * qi for ki, qi in zip(self.kvec, q_k))
        # q_perp = q - khat (khat . q) = q - k (k . q) / k^2   (note: k^2, not k)
        k2_safe = self.k2.copy(); k2_safe[0, 0, 0] = 1.0       # avoid 0/0 at k=0
        perp_sq = sum(np.abs(qi - ki * k_dot_q / k2_safe) ** 2 for ki, qi in zip(self.kvec, q_k))
        #total_sq = sum(np.abs(qi) ** 2 for qi in q_k)
        kmag = np.sqrt(self.k2)
        norm = L**3 / N**6
        k, P_perp = self.shell_average(perp_sq, kmag, edges, norm)
        #_, P_total = self.shell_average(total_sq, kmag, edges, norm)
        return k, P_perp

    def scalar_power(self, chi, L, edges):
        """Power spectrum P_chi(k) of a real scalar field (here chi = x_e (1+delta_L))."""
        N = chi.shape[-1]
        return self.shell_average(np.abs(np.fft.rfftn(chi))**2, np.sqrt(self.k2), edges, L**3 / N**6)


    def v2_missing(self, CosmoParams, z, k_box, Npoints=200, kmin_tol=2e-2, extend_below_kmin=False):
        """<v^2>_miss / c^2: variance of the velocity carried by modes with k < k_box (spec eq. 7).
        Velocity variance of a Gaussian field: <v^2> = (aHf)^2 ∫ P(k) dk / (2 pi^2) (3-D rms squared).
        P_delta(k,z) already includes the growth D(z)^2, so no extra D(z)/D(z0) factor.
        """
        # TODO: can we instead use the pre-computed CosmoParams._klistCF, CosmoParams._PklinCF and rescale with (D(z)/D(0))²?
        #k_list = np.logspace(-6, np.log10(k_box), 2000)
        kmin = kmin_tol*k_box
        k  = np.logspace(np.log10(kmin), np.log10(k_box), Npoints)  # TODO: test optimal values of kmin and Npoints; discuss whether we should use CosmoParams._klistCF.min() here
        P_k = np.array([CosmoParams.ClassCosmo.pk(ki, z) for ki in k])   # TODO: for linear theory, we could optimize as P_z = P_z0 * (CosmoParams.growthint(z) / CosmoParams.growthint(self.z0))**2 instead of recomputing at each z
        integral = np.trapezoid(P_k / (2 * np.pi**2), k)
        if extend_below_kmin:   # For analytical approximation below kmin: ∫_0^kmin P(kmin)(k/kmin)^ns dk/(2π²)
            integral += P_k[0] * kmin / (1 + CosmoParams.ns) / (2*np.pi**2)
        a_H_f = (1./(1. + z)) * cosmology.Hub(CosmoParams, z) * self.growth_ratio_logderivative(CosmoParams, z)   # [km/s/Mpc]
        return a_H_f**2 * integral / constants.c_kms**2   # dimensionless (units of c^2)


    @staticmethod
    def He_factor(CosmoParams, z):
        """Electrons per hydrogen atom for a fully ionized H+He gas: 1+x_He (He II) above z=3, 1+2x_He (He III) below."""
        return 1.0 + CosmoParams.x_He + CosmoParams.x_He * (z < constants.zHeIIreio)

    def tau_of_z(self, CosmoParams, z, zgrid, xbar, nlow=400):
        """Optical depth tau(zeval) from z=0 (spec eq. 10). xbar = mass-weighted ionized fraction on zgrid.
        Pieces: (i) 0 < z < zgrid.min(): x=1;  (ii) inside the grid: the box's own xbar(z);  (iii) above the grid: x=0.
        """
        pref = constants.sigmaT * constants.Mpctocm * constants.c_kms * cosmology.n_H(CosmoParams, 0.)   # sigma_T c n_H0 in [Mpc * (1/cm^2 * cm^-3 ...)] --> dimensionless with dz/H
        zl = np.linspace(0.0, zgrid.min(), nlow)
        tau_low = pref * np.trapezoid((1 + zl) ** 2 * self.He_factor(CosmoParams, zl) / cosmology.Hub(CosmoParams, zl), zl)
        integrand = pref * (1 + zgrid) ** 2 * self.He_factor(CosmoParams, zgrid) * xbar / cosmology.Hub(CosmoParams, zgrid)
        cumulative = np.concatenate([[0.0], np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(zgrid))])
        return tau_low + np.interp(z, zgrid, cumulative)


    @staticmethod
    def loglog_interp(kq, k, P):
        """Interpolate P at kq in log k - log P. Raises error if kq is outside the resolved band (no silent clipping outside the boundaries)."""
        if kq < k.min() or kq > k.max():
            raise ValueError(f"k = {kq:.3f} Mpc^-1 outside the resolved band [{k.min():.3f}, {k.max():.3f}]")
        positive = P > 0   # At very high z there may be no ionized cell: P=0
        if positive.sum() < 2:
            return 0.
        return float(np.exp(np.interp(np.log(kq), np.log(k[positive]), np.log(P[positive]))))

    def limber(self, CosmoParams, zgrid, ell_list, k_list, P_list, tau):
        """Limber integral. P_list[i] = P_q⊥(k) (units (v/c)^2 Mpc^3) at zgrid[i], tabulated at k_list[i].
        Returns D_l [microK^2] for each ell and the integrand of D_l (contribution per unit z), used for the cumulative D_l(>z).
        """
        s = cosmology.chi_of_redshift(CosmoParams, zgrid)   # comoving distance [Mpc]
        a = 1.0 / (1.0 + zgrid)
        ne0 = cosmology.n_H(CosmoParams, 0.) * (1.0 + CosmoParams.x_He)  # electrons per comoving volume if H and He are both singly ionized, starting from the He and H number density today (spec eq. 9)
        prefactor = (constants.sigmaT * constants.Mpctocm * ne0) ** 2   # (sigma_T n_e0)^2 [Mpc^-2]; q is in units of c, so no 1/c^2 here
        ds_dz = constants.c_kms / cosmology.Hub(CosmoParams, zgrid)   # ds = (c/H) dz [Mpc]
        D_ell, contributions = [], []
        for ell in ell_list:
            P_at_k = np.array([self.loglog_interp(ell / s[i], k_list[i], P_list[i]) for i in range(len(zgrid))])
            integrand_z = prefactor * ds_dz / (s**2 * a**4) * np.exp(-2.0 * tau) * P_at_k / 2.0      # dC_ell/dz
            to_D = ell * (ell + 1) * CosmoParams.ClassCosmo.T_cmb()**2 / (2*np.pi)*1e12  # -> microK^2
            D_ell.append(to_D * np.trapezoid(integrand_z, zgrid))
            contributions.append(to_D * integrand_z)
        return np.array(D_ell), np.array(contributions)