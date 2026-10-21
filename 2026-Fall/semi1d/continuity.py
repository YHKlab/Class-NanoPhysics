"""
semi1d.continuity: continuity equations with generation and recombination (Neamen Ch. 6).

Module of Labs 5 and 6; built on semi1d.drift_diffusion (the divergence of the drift-diffusion current).

    (1/q) dJ_p/dx = G - R,     (1/q) dJ_n/dx = R - G          (steady state)
    SRH  R = (np - n_i^2) / [tau_p0 (n + n') + tau_n0 (p + p')],  n' = p' = n_i  (single midgap trap,
         Neamen (6.99), Sec. 6.5)

Ends: an ohmic contact holds the equilibrium (or injected) density, a free end has no current through the
surface (no surface recombination, Neamen Sec. 6.6 with s = 0). Textbook results: diffusion lengths,
low-injection excess-carrier profiles, the uniform SRH steady state, and the ambipolar coefficients
D' (6.43) and mu' (6.41).

Device namespace used here: as in semi1d.drift_diffusion, plus ni, taun, taup [s], srh (False: R = 0).
"""

import numpy as np

from .drift_diffusion import flux_operator, solve_with_ends


###############################################################################
# Generation and Recombination
###############################################################################

def srh_denominator(dev, n, p):
    """tau_p (n + n_i) + tau_n (p + n_i) [s cm^-3]; infinite with SRH off (R = 0)."""
    if not dev.srh:
        return np.full(np.shape(n), np.inf)
    return dev.taup * (n + dev.ni) + dev.taun * (p + dev.ni)


def srh_rate(dev, n, p):
    """Net SRH recombination rate R [cm^-3 s^-1] (midgap trap, n' = p' = n_i), Neamen (6.99)."""
    return (n * p - dev.ni ** 2) / srh_denominator(dev, n, p)


def generation_profile(x_um, G, depth_um=0.0):
    """Generation rate on the grid [cm^-3 s^-1]: uniform G, or G exp(-s/d) with the absorption depth d = depth_um
    measured from the left end."""
    if depth_um > 0:
        return G * np.exp(-np.asarray(x_um) / depth_um)
    return np.full(np.shape(x_um), float(G))


###############################################################################
# Continuity Equation
###############################################################################

def solve_continuity(dev, psi, n, p, G, carrier):
    """
    Steady-state continuity equation of one carrier with psi fixed (linear, tridiagonal): the drift-diffusion
    flux operator plus G - R, with SRH linearized at fixed denominator, R = other*own/den - n_i^2/den.

    input:
        np.array    psi, n, p       Potential [V] and current concentrations [cm^-3]
        np.array    G               Generation rate [cm^-3 s^-1]
        str         carrier         'n' (electrons) or 'p' (holes)
    output:
        np.array                    New concentration of the carrier [cm^-3]
    """
    lower, diag, upper, free_l, free_r, sign = flux_operator(dev, psi, carrier)
    own, other = (p, n) if carrier == 'p' else (n, p)
    den = srh_denominator(dev, n, p)
    diag += sign * other / den
    rhs = sign * (G + dev.ni ** 2 / den)
    return solve_with_ends(dev, lower, diag, upper, rhs, own, free_l, free_r)


###############################################################################
# Textbook Results of Chapter 6
###############################################################################

def diffusion_length(D, tau):
    """Diffusion length L = (D tau)^{1/2} [cm]."""
    return np.sqrt(D * tau)


def D_ambipolar(n, p, Dn, Dp):
    """Ambipolar diffusion coefficient D' = D_n D_p (n + p) / (D_n n + D_p p), Neamen (6.43)."""
    return Dn * Dp * (n + p) / (Dn * n + Dp * p)


def mu_ambipolar(n, p, mun, mup):
    """Ambipolar mobility mu' = mu_n mu_p (p - n) / (mu_n n + mu_p p), Neamen (6.41)."""
    return mun * mup * (p - n) / (mun * n + mup * p)


def delta_uniform_srh(G, n0, p0, ni, taun, taup):
    """Uniform steady state R_SRH(n0 + delta, p0 + delta) = G at any injection level (n' = p' = n_i) [cm^-3]."""
    b = n0 + p0 - G * (taun + taup)
    c = -G * (taup * (n0 + ni) + taun * (p0 + ni))
    disc = np.sqrt(b * b - 4 * c)
    return (-b + disc) / 2 if b < 0 else -2 * c / (b + disc)


def finite_decay(x, L_bar, L_diff):
    """sinh((L_bar - x)/L)/sinh(L_bar/L), written without overflow; 1 - x/L_bar without recombination."""
    if not np.isfinite(L_diff):
        return 1.0 - np.asarray(x) / L_bar
    return (np.exp(-x / L_diff) - np.exp(-(2 * L_bar - x) / L_diff)) / (1 - np.exp(-2 * L_bar / L_diff))


def excess_profile(s, W, G0, D, tau, depth=0.0, left='ohmic', right='ohmic', delta0=0.0):
    """
    Low-injection excess carriers of a bar 0 <= s <= W [cm]: D delta'' - delta/tau = -G(s) with G = G0 (depth = 0)
    or G0 exp(-s/depth). Ends: 'ohmic' delta = 0 (left contact: delta = delta0), 'free' delta' = 0 (no surface
    recombination). tau = inf: no recombination. nan when no steady state exists (both ends free, no recombination).
    """
    s = np.asarray(s, dtype=float)
    if np.isfinite(tau):
        Ld = np.sqrt(D * tau)
        if depth > 0 and abs(depth - Ld) < 1e-6 * Ld:
            depth = Ld * (1 + 1e-6)                              # d = L: limit of the general solution

        def basis(u):                                            # (h1, h2, particular) and their derivatives
            h = (np.exp(-u / Ld), np.exp((u - W) / Ld))
            dh = (-h[0] / Ld, h[1] / Ld)
            if depth > 0:
                P = G0 * tau * depth ** 2 / (depth ** 2 - Ld ** 2) * np.exp(-u / depth)
                return h, dh, P, -P / depth
            return h, dh, np.full_like(u, G0 * tau), np.zeros_like(u)
    else:
        def basis(u):
            h, dh = (np.ones_like(u), u / W), (np.zeros_like(u), np.full_like(u, 1.0 / W))
            if depth > 0:
                P = -G0 * depth ** 2 / D * np.exp(-u / depth)
                return h, dh, P, -P / depth
            return h, dh, -G0 * u ** 2 / (2 * D), -G0 * u / D
    h, dh, P, dP = basis(np.array([0.0, W]))
    A, b = np.zeros((2, 2)), np.zeros(2)
    for r, (bc, target) in enumerate(((left, delta0), (right, 0.0))):
        if bc == 'free':
            A[r], b[r] = (dh[0][r], dh[1][r]), -dP[r]
        else:
            A[r], b[r] = (h[0][r], h[1][r]), target - P[r]
    if abs(np.linalg.det(A)) < 1e-300:
        return np.full_like(s, np.nan)
    c = np.linalg.solve(A, b)
    h, _, P, _ = basis(s)
    return P + c[0] * h[0] + c[1] * h[1]
