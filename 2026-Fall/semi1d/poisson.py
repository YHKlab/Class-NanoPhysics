"""
semi1d.poisson: the Poisson equation for the electrostatic potential (Neamen Ch. 7).

Module of Lab 6 (Labs 4 and 5 use quasi-neutrality instead, as the textbook does in Chs. 5 and 6).

    d^2 psi/dx^2 = -(q/eps) (p - n + N_d - N_a),     E = -d psi/dx          Neamen (7.11)

Carriers follow Boltzmann statistics (semi1d.statistics) with frozen quasi-Fermi levels, so the equation is
nonlinear in psi; it is solved by a damped Newton iteration with a tridiagonal Jacobian. Ohmic ends keep psi
fixed; free ends have no field outside the surface (d psi/dx = 0). Textbook results of the abrupt junction in
the depletion approximation: V_bi (7.10), x_n (7.28), x_p (7.29), W (7.31), (7.34), |E|max (7.36).

Device namespace used here: x, dx, N, C (= N_d - N_a), eps, Vt, ni, bc_left, bc_right.
"""

import numpy as np
from scipy.linalg import solve_banded
from scipy import constants as sc_const

from .statistics import neutral_potential, boltzmann_densities

Q = sc_const.e          # Elementary charge [C]


def span(dev):
    """Unknown nodes lo:hi: ohmic end nodes are fixed, free end nodes are solved."""
    return (0 if dev.bc_left == 'free' else 1), (dev.N if dev.bc_right == 'free' else dev.N - 1)


def solve_poisson(dev, psi, n_ref, p_ref, psi_ref, maxit=200, tol=1e-11):
    """
    Nonlinear Poisson equation with n = n_ref exp((psi - psi_ref)/V_t) and p = p_ref exp(-(psi - psi_ref)/V_t)
    (quasi-Fermi levels frozen at their values for psi_ref).

    input:
        np.array    psi             Initial potential [V] (end values at ohmic contacts are kept)
        np.array    n_ref, p_ref    Concentrations at psi_ref [cm^-3]
        np.array    psi_ref         Reference potential [V]
        int, float  maxit, tol      Newton iterations and tolerance [V]
    output:
        np.array    psi             Potential [V]
    """
    Vt, a, C = dev.Vt, Q * dev.dx ** 2 / dev.eps, dev.C
    # Ohmic end: psi fixed. Free end: no field outside the surface (mirror node psi[-1] = psi[1]).
    lo, hi = span(dev)
    upper, lower = np.ones(dev.N), np.ones(dev.N)
    upper[0] = lower[-1] = 2.0
    ab = np.zeros((3, hi - lo))
    ab[0, 1:], ab[2, :-1] = upper[lo:hi - 1], lower[lo + 1:hi]
    psi = psi.copy()
    lap = np.zeros(dev.N)
    for _ in range(maxit):
        u = np.clip((psi - psi_ref) / Vt, -80, 80)
        nn = n_ref * np.exp(u)          # n = n_ref exp((psi - psi_ref)/V_t)
        pp = p_ref * np.exp(-u)         # p = p_ref exp(-(psi - psi_ref)/V_t)
        lap[1:-1] = psi[2:] - 2 * psi[1:-1] + psi[:-2]
        lap[0], lap[-1] = 2 * (psi[1] - psi[0]), 2 * (psi[-2] - psi[-1])
        F = (lap + a * (pp - nn + C))[lo:hi]
        ab[1, :] = (-2.0 - a * (pp + nn) / Vt)[lo:hi]
        dpsi = solve_banded((1, 1), ab, -F)
        dpsi = np.sign(dpsi) * Vt * np.log1p(np.abs(dpsi) / Vt)   # Damp large Newton steps
        psi[lo:hi] += dpsi
        if np.max(np.abs(dpsi)) < tol:
            break
    return psi


def equilibrium(dev, maxit=200, tol=1e-11):
    """
    Equilibrium solution (Poisson-Boltzmann equation, E_F = 0 everywhere), starting from local charge
    neutrality psi = V_t asinh[(N_d - N_a)/2n_i] (semi1d.statistics).

    output:
        np.array    psi_eq, n_eq, p_eq      Potential [V] and concentrations [cm^-3]
    """
    psi = neutral_potential(dev.C, dev.ni, dev.Vt)
    ref = np.full(dev.N, dev.ni)
    psi_eq = solve_poisson(dev, psi, ref, ref, np.zeros(dev.N), maxit, tol)
    n_eq, p_eq = boltzmann_densities(psi_eq, dev.ni, dev.Vt)
    return psi_eq, n_eq, p_eq


def space_charge(dev, n, p):
    """Net charge density divided by q, rho/q = p - n + N_d - N_a [cm^-3]."""
    return p - n + dev.C


def electric_field(dev, psi):
    """Electric field E = -d psi/dx [V/cm] (central differences)."""
    return np.gradient(-psi, dev.x)


###############################################################################
# Textbook Results of Chapter 7 (abrupt junction, depletion approximation)
###############################################################################

def debye_length(N, eps, Vt):
    """Extrinsic Debye length L_D = (eps kT / e^2 N)^{1/2} [cm]."""
    return np.sqrt(eps * Vt / (Q * N))


def built_in_potential(Na, Nd, ni, Vt):
    """Built-in potential V_bi = (kT/e) ln(N_a N_d / n_i^2) [V], Neamen (7.10)."""
    return Vt * np.log(Na * Nd / ni ** 2)


def depletion_width(Na, Nd, Vtot, eps):
    """Depletion width W = [2 eps (V_bi + V_R)/e (N_a + N_d)/(N_a N_d)]^{1/2} [cm], Neamen (7.31), (7.34);
    x_n = W N_a/(N_a + N_d) (7.28), x_p = W N_d/(N_a + N_d) (7.29). Vtot = V_bi - V."""
    return np.sqrt(2 * eps * Vtot / Q * (Na + Nd) / (Na * Nd))


def max_field(Na, Nd, Vtot, eps):
    """Maximum field |E|max = [2e (V_bi + V_R)/eps N_a N_d/(N_a + N_d)]^{1/2} [V/cm], Neamen (7.36)."""
    return np.sqrt(2 * Q * Vtot / eps * Na * Nd / (Na + Nd))


def depletion_profiles(xc_cm, Na, Nd, xp_cm, xn_cm, psi_p, eps):
    """
    Depletion-approximation rho/q [cm^-3], E [V/cm] and psi [V] of the abrupt junction (Neamen (7.12)-(7.25)).
    xc_cm: position from the metallurgical junction [cm]; xp_cm, xn_cm: depletion edges [cm]; psi_p: potential of
    the neutral p region [V].
    """
    xc, a, b = np.asarray(xc_cm, dtype=float), xp_cm, xn_cm
    rho = np.where((xc > -a) & (xc < 0), -Na, 0.0) + np.where((xc >= 0) & (xc < b), Nd, 0.0)
    E = np.where((xc > -a) & (xc < 0), -Q * Na / eps * (xc + a), 0.0) \
        + np.where((xc >= 0) & (xc < b), -Q * Nd / eps * (b - xc), 0.0)
    Vj = Q * Na * a ** 2 / (2 * eps) + Q * Nd * b ** 2 / (2 * eps)
    psi = np.where(xc <= -a, psi_p, 0.0) \
        + np.where((xc > -a) & (xc < 0), psi_p + Q * Na / (2 * eps) * (xc + a) ** 2, 0.0) \
        + np.where((xc >= 0) & (xc < b), psi_p + Vj - Q * Nd / (2 * eps) * (b - xc) ** 2, 0.0) \
        + np.where(xc >= b, psi_p + Vj, 0.0)
    return rho, E, psi
