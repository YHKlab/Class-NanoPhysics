"""
semi1d.statistics: carrier statistics in and near thermal equilibrium (Neamen Ch. 3-4).

Module of Labs 4, 5 and 6. Every density in the labs comes from here, so the statistical assumptions of the
whole simulator are made in this file:

    Boltzmann approximation   f_F(E) ~ exp[-(E - E_F)/kT]                      Neamen (3.80)
                              n = n_i exp[(E_F - E_Fi)/kT],  p = n_i exp[(E_Fi - E_F)/kT]  (nondegenerate)
    Complete ionization       N_d+ = N_d, N_a- = N_a                           Neamen Sec. 4.4.2
    Charge neutrality         n0 + N_a = p0 + N_d                              Neamen (4.56), (4.60)
    n_i(T)                    n_i(300 K) (T/300)^{3/2} exp[-(E_g/2k)(1/T - 1/300)] with a constant E_g

Out of equilibrium the same relations hold with the quasi-Fermi levels E_Fn and E_Fp (Neamen Sec. 6.4).
Energies in eV, potentials in V (E_Fi = -q psi), densities in cm^-3.
"""

import numpy as np
from scipy import constants as sc_const

KB_EV = sc_const.physical_constants["Boltzmann constant in eV/K"][0]     # [eV/K]


def thermal_voltage(T):
    """kT/e [V], scaled from 0.0259 V at 300 K."""
    return 0.0259 * T / 300.0


def intrinsic_density(T, Eg, ni300):
    """Intrinsic concentration [cm^-3]: n_i(300 K) (T/300)^{3/2} exp[-(E_g/2k)(1/T - 1/300)], constant E_g."""
    return ni300 * (T / 300.0) ** 1.5 * np.exp(-Eg / (2 * KB_EV) * (1.0 / T - 1.0 / 300.0))


def equilibrium_densities(Na, Nd, ni):
    """
    Thermal-equilibrium n0, p0 [cm^-3] with complete ionization and charge neutrality, Neamen (4.60), (4.62):
    n0 = (N_d - N_a)/2 + [((N_d - N_a)/2)^2 + n_i^2]^{1/2},  p0 = n_i^2 / n0  (written stably for both signs).
    """
    net = np.asarray(Nd, dtype=float) - np.asarray(Na, dtype=float)
    root = np.sqrt((net / 2) ** 2 + ni ** 2)
    n0 = np.where(net >= 0, net / 2 + root, ni ** 2 / (-net / 2 + root))
    return n0, ni ** 2 / n0


def neutral_potential(C, ni, Vt):
    """Potential of a charge-neutral region, psi = V_t asinh[(N_d - N_a)/2n_i] [V] (E_F = 0); equivalent to (4.60)."""
    return Vt * np.arcsinh(C / (2 * ni))


def boltzmann_densities(psi, ni, Vt, phi_n=0.0, phi_p=0.0):
    """n = n_i exp[(psi - phi_n)/V_t], p = n_i exp[(phi_p - psi)/V_t] [cm^-3]; phi_n, phi_p = quasi-Fermi potentials
    (zero at equilibrium)."""
    return ni * np.exp((psi - phi_n) / Vt), ni * np.exp((phi_p - psi) / Vt)


def band_diagram(psi, n, p, ni, Vt, Eg):
    """
    Band edges and quasi-Fermi levels [eV] (energy zero = equilibrium Fermi level, E_Fi = -q psi):
    E_c, E_v = E_Fi +- E_g/2 (midgap E_Fi), E_Fn = E_Fi + kT ln(n/n_i), E_Fp = E_Fi - kT ln(p/n_i).
    """
    Ei = -psi
    return dict(Ec=Ei + Eg / 2, Ev=Ei - Eg / 2, Ei=Ei,
                Fn=Ei + Vt * np.log(np.maximum(n, 1e-300) / ni),
                Fp=Ei - Vt * np.log(np.maximum(p, 1e-300) / ni))


def fermi_level(n0, ni, Vt):
    """E_F - E_Fi = kT ln(n0/n_i) [eV] (Boltzmann approximation)."""
    return Vt * np.log(n0 / ni)
