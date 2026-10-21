"""
semi1d.drift_diffusion: carrier transport by drift and diffusion (Neamen Ch. 5).

Module of Labs 4, 5 and 6.

    J_n = q mu_n n E + q D_n dn/dx,   J_p = q mu_p p E - q D_p dp/dx,   E = -d psi/dx
    D/mu = kT/q  (Einstein relation, Neamen (5.47); nondegenerate statistics)

Numerics: Scharfetter-Gummel discretization between grid nodes (exact for a constant field between nodes) and
the tridiagonal operator of the current divergence used by semi1d.continuity. Textbook results: the induced field of a graded doping in equilibrium
(Neamen (5.42), quasi-neutral n ~ N_d(x), (5.39)) and the conductivity sigma = e(mu_n n + mu_p p).

Device namespace used here: x, dx, N (nodes), Vt, mun, mup, Dn, Dp, bc_left, bc_right ('ohmic' or 'free').
"""

import numpy as np
from types import SimpleNamespace
from scipy.linalg import solve_banded
from scipy import constants as sc_const

Q = sc_const.e          # Elementary charge [C]


###############################################################################
# Relations of Chapter 5
###############################################################################

def einstein(mu, Vt):
    """Diffusion coefficient D = mu kT/q [cm^2/s] (Einstein relation, Neamen (5.47))."""
    return mu * Vt


def conductivity(n, p, mun, mup):
    """Conductivity sigma = e(mu_n n + mu_p p) [S/cm]."""
    return Q * (mun * n + mup * p)


def induced_field(N, dNdx, Vt, doping_type='n'):
    """Induced field of a graded doping in equilibrium [V/cm], quasi-neutral (5.39):
    n-type E = -(kT/e)(1/N_d)(dN_d/dx) (5.42); p-type E = +(kT/e)(1/N_a)(dN_a/dx)."""
    sign = -1.0 if doping_type == 'n' else 1.0
    return sign * Vt * dNdx / N


def quasi_neutral_field(dev, n, p, dndx, dpdx, J=0.0):
    """
    Field that carries a uniform total current J [A/cm^2] through given densities (quasi-neutral model):
        J = q(mu_n n + mu_p p) E + q(D_n dn/dx - D_p dp/dx)
    J = 0 gives the equilibrium (induced) field; for n = N_d(x), p ~ 0 it is (5.42).
    """
    return (J / Q - (dev.Dn * dndx - dev.Dp * dpdx)) / (dev.mun * n + dev.mup * p)


###############################################################################
# Scharfetter-Gummel Currents
###############################################################################

def bernoulli(x):
    """Bernoulli function B(x) = x / (e^x - 1) (Scharfetter-Gummel weights)."""
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    small = np.abs(x) < 1e-4
    xs = x[small]
    out[small] = 1.0 - xs / 2.0 + xs ** 2 / 12.0
    xl = x[~small]
    with np.errstate(over='ignore'):
        out[~small] = xl / np.expm1(xl)
    return out


def currents(dev, psi, n, p):
    """
    Current densities between nodes (half-grid points) [A/cm^2]: Scharfetter-Gummel totals, the drift part from
    the average concentration and the midpoint field, and diffusion as the remainder.

    output:
        SimpleNamespace     Jn, Jp, Jn_drift, Jp_drift, Jn_diff, Jp_diff, Jt (= Jn + Jp)
    """
    dl = (psi[1:] - psi[:-1]) / dev.Vt
    Bp, Bm = bernoulli(dl), bernoulli(-dl)
    Jn = Q * dev.Dn / dev.dx * (Bp * n[1:] - Bm * n[:-1])
    Jp = Q * dev.Dp / dev.dx * (Bp * p[:-1] - Bm * p[1:])
    Eh = -(psi[1:] - psi[:-1]) / dev.dx
    Jn_drift = Q * dev.mun * 0.5 * (n[1:] + n[:-1]) * Eh
    Jp_drift = Q * dev.mup * 0.5 * (p[1:] + p[:-1]) * Eh
    return SimpleNamespace(Jn=Jn, Jp=Jp, Jn_drift=Jn_drift, Jp_drift=Jp_drift, Jn_diff=Jn - Jn_drift,
                           Jp_diff=Jp - Jp_drift, Jt=Jn + Jp)


###############################################################################
# Current-Divergence Operator
###############################################################################

def span(dev):
    """Unknown nodes lo:hi of the linear systems: ohmic end nodes are fixed, free end nodes are solved."""
    return (0 if dev.bc_left == 'free' else 1), (dev.N if dev.bc_right == 'free' else dev.N - 1)


def flux_operator(dev, psi, carrier):
    """
    Tridiagonal Scharfetter-Gummel operator of the particle-current divergence of one carrier, with psi fixed:
        holes     (D/dx^2)[B(D_i)p_i - B(-D_i)p_i+1 - B(D_i-1)p_i-1 + B(-D_i-1)p_i]   (= (1/q) dJ_p/dx)
        electrons (D/dx^2)[B(D_i)n_i+1 - B(-D_i)n_i - B(D_i-1)n_i + B(-D_i-1)n_i-1]   (= (1/q) dJ_n/dx)
    Interior rows only; free-end rows (half cell, no current through the surface) are returned separately.

    output:
        lower, diag, upper      Coefficients on all nodes (end rows zero)
        free_l, free_r          (diag, upper) of node 0 and (lower, diag) of node N-1 for free ends
        sign                    +1 for holes, -1 for electrons (sign of G - R on the right-hand side)
    """
    dl = (psi[1:] - psi[:-1]) / dev.Vt
    Bp, Bm = bernoulli(dl), bernoulli(-dl)
    lower, diag, upper = np.zeros(dev.N), np.zeros(dev.N), np.zeros(dev.N)
    if carrier == 'p':
        D = dev.Dp / dev.dx ** 2
        lower[1:-1], upper[1:-1] = -D * Bp[:-1], -D * Bm[1:]
        diag[1:-1] = D * (Bp[1:] + Bm[:-1])
        return lower, diag, upper, (2 * D * Bp[0], -2 * D * Bm[0]), (-2 * D * Bp[-1], 2 * D * Bm[-1]), 1.0
    D = dev.Dn / dev.dx ** 2
    lower[1:-1], upper[1:-1] = D * Bm[:-1], D * Bp[1:]
    diag[1:-1] = -D * (Bm[1:] + Bp[:-1])
    return lower, diag, upper, (-2 * D * Bm[0], 2 * D * Bp[0]), (2 * D * Bm[-1], -2 * D * Bp[-1]), -1.0


def solve_with_ends(dev, lower, diag, upper, rhs, own, free_l, free_r):
    """
    Solve the tridiagonal system with the end conditions of the device. Ohmic ends keep own[0], own[-1] and are
    removed from the system (their coupling moves to the right-hand side; a unit Dirichlet row next to rows of
    ~D would trigger pivoting and spoil small minority densities). Free ends use the half-cell rows.
    """
    lo, hi = span(dev)
    if lo == 0:
        diag[0] += free_l[0]
        upper[0] = free_l[1]
    else:
        rhs[1] -= lower[1] * own[0]
    if hi == dev.N:
        lower[-1] = free_r[0]
        diag[-1] += free_r[1]
    else:
        rhs[-2] -= upper[-2] * own[-1]
    ab = np.zeros((3, hi - lo))
    ab[0, 1:], ab[1], ab[2, :-1] = upper[lo:hi - 1], diag[lo:hi], lower[lo + 1:hi]
    out = own.copy()
    out[lo:hi] = solve_banded((1, 1), ab, rhs[lo:hi])
    return out
