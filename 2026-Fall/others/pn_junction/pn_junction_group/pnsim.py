# %%
#!/usr/bin/env python3
"""
Simplified semiconductor device simulator using NumPy arrays and SciPy constants.
• Duplicated grid-size constants have been eliminated.
• The unit conversions use literal values.
• Only essential widgets are provided (a few parameter inputs and a Run button).
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from ipywidgets import HBox, VBox
from IPython.display import display, clear_output
import scipy.constants as const

# Constants
EPSILON_0 = const.epsilon_0       # F/m
K_B       = const.k               # J/K
E_CHARGE  = const.e               # C
HP        = const.h               # J s

# -----------------------
# Simulator Class
# -----------------------
class DeviceSimulator:
    def __init__(self, params):
        self.params = params
        self.Nx = params['Nx']
        self.N_points = self.Nx + 2
        self.Temp = params['Temp']
        self.setup_mesh()
        self.ep = 11.7 * EPSILON_0
        self.Ut = 0.0259

        # Allocate arrays
        self.Voltage = np.zeros(self.N_points)
        self.E_field = np.zeros(self.N_points)
        self.p = np.zeros(self.N_points)
        self.n = np.zeros(self.N_points)
        self.Dop   = np.zeros(self.N_points)
        self.Dop_x = np.zeros(self.N_points)
        self.Dop_y = np.zeros(self.N_points)
        self.p_eq  = np.zeros(self.N_points)
        self.n_eq  = np.zeros(self.N_points)
        
        self.initialize_doping()
        self.initialize_equilibrium()

    def setup_mesh(self):
        xi, xf = 0.0, self.params['semi_length']
        self.dx = (xf - xi) / (self.Nx + 1)
        self.x = np.linspace(xi, xf, self.N_points)

    def initialize_doping(self):
        self.params['Na_profile'] = np.array(self.params['Na_profile'], dtype=float)
        self.params['Nd_profile'] = np.array(self.params['Nd_profile'], dtype=float)
        self.Dop   = self.params['Na_profile'] - self.params['Nd_profile']
        self.Dop_x = self.params['Na_profile']
        self.Dop_y = self.params['Nd_profile']

    def initialize_equilibrium(self):
        ni = self.params['ni']
        self.eg = self.params['Eg']
        self.p_eq = 0.5 * (np.sqrt(self.Dop**2 + 4*ni**2) + self.Dop)
        self.n_eq = 0.5 * (np.sqrt(self.Dop**2 + 4*ni**2) - self.Dop)

    def initialize_boundary_conditions(self, v_applied):
        ni, Ut = self.params['ni'], self.Ut
        mid = self.N_points // 2

        # Fermi level at contacts (벡터화 버전)
        def fermi_lvl(n, p, v):
            # n, p가 배열이므로 np.where를 써서 요소별 조건부 계산 수행
            # n > p 이면 Ut * log(n/ni) + v, 그렇지 않으면 -Ut * log(p/ni) + v
            return np.where(
                n > p,
                Ut * np.log(n / ni) + v,
                -Ut * np.log(p / ni) + v
            )

        self.p = self.p_eq
        self.n = self.n_eq

        if self.params['F_scatt']:
            self.G = np.zeros(self.N_points)
            if self.params['g_type'] == 'uniform':
                self.G.fill(self.params['g_density'])
            elif self.params['g_type'] == 'end':
                self.G[0] = self.params['g_density']  # 0:1 대신 0만 사용
            self.n += self.G
            self.p += self.G

        linV = np.linspace(v_applied, 0, self.N_points)
        Volt = fermi_lvl(self.n, self.p, 0)  # 배열 단위로 계산
        self.Voltage = Volt + linV

        # Dirichlet BC on Voltage
        # Initial E‐field
        self.E_field[1:-1] = -(self.Voltage[2:] - self.Voltage[:-2]) / (2 * self.dx)

    @staticmethod
    def bernoulli(x, tol=1e-7):
        if x > 50:
            x = 50
        if abs(x) < tol:
            return 1 - x/2 + x*x/12 - x**4/720
        return x / (math.exp(x) - 1)

    def tdm_solve(self, a, b, c, r, n_internal):
        beta = np.zeros(n_internal+2)
        rho  = np.zeros(n_internal+2)
        x_sol = np.zeros(n_internal+2)
        beta[1] = b[1]
        rho[1]  = r[1]
        for j in range(2, n_internal+1):
            beta[j] = b[j] - a[j]*c[j-1]/beta[j-1]
            rho[j]  = r[j] - a[j]*rho[j-1]/beta[j-1]
        x_sol[n_internal] = rho[n_internal]/beta[n_internal]
        for j in range(n_internal-1, 0, -1):
            x_sol[j] = (rho[j] - c[j]*x_sol[j+1]) / beta[j]
        return x_sol

    def solve_poisson(self, tol=1e-7):
        n_int, dx, Ut = self.N_points-2, self.dx, self.Ut
        ec, ep = E_CHARGE, self.ep
        a = np.zeros(self.N_points)
        b = np.zeros(self.N_points)
        c = np.zeros(self.N_points)
        r = np.zeros(self.N_points)
        oldV = self.Voltage.copy()
        # Define max exponent magnitude to avoid overflow
        max_exp = 50.0

        while True:
            a.fill(0); b.fill(0); c.fill(0); r.fill(0)
            for i in range(1, self.N_points-1):
                # safe exponent arguments
                arg_p = (oldV[i] - self.Voltage[i]) / Ut
                arg_n = (self.Voltage[i] - oldV[i]) / Ut
                arg_p = max(-max_exp, min(max_exp, arg_p))
                arg_n = max(-max_exp, min(max_exp, arg_n))
                exp_p = math.exp(arg_p)
                exp_n = math.exp(arg_n)
                a[i] = 1.0
                b[i] = -2 - (ec/ep)*dx*dx/Ut * (self.p[i]*exp_p + self.n[i]*exp_n)
                c[i] = 1.0
                r[i] = -(
                    self.Voltage[i+1] + self.Voltage[i-1] - 2*self.Voltage[i]
                ) - (ec/ep)*dx*dx*(
                    self.p[i]*exp_p - self.n[i]*exp_n - self.Dop[i]
                )
            delta = self.tdm_solve(a, b, c, r, n_int)
            diff = np.sum(np.abs(delta[1:-1]))
            self.Voltage[1:-1] += delta[1:-1]
            if diff <= tol:
                break

        # update E‐field
        self.E_field[1:-1] = -(self.Voltage[2:] - self.Voltage[:-2])/(2*dx)

    def solve_continuity(self, tol_abs, tol_rel=1e-6):
        n_int, dx, Ut = self.N_points-2, self.dx, self.Ut
        mu_p = np.ones(self.N_points)*self.params['mu_p_P']
        mu_n = np.ones(self.N_points)*self.params['mu_n_P']
        Dp, Dn = mu_p*Ut, mu_n*Ut
        tau_p, tau_n = self.params['tau_p'], self.params['tau_n']
        ni, F_scatt = self.params['ni'], self.params['F_scatt']
        a = np.zeros(self.N_points)
        b = np.zeros(self.N_points)
        c = np.zeros(self.N_points)
        r = np.zeros(self.N_points)

        # Holes
        while True:
            a.fill(0); b.fill(0); c.fill(0); r.fill(0)
            for i in range(1, self.N_points-1):
                RecombRate = 1.0/(tau_n*(self.p[i]) + tau_p*(self.n[i]))
                co0 = Dp[i]/(dx*dx)
                co1 = Dp[i+1]/(dx*dx)
                co_a = co0 * self.bernoulli((self.Voltage[i]-self.Voltage[i-1])/Ut)
                co_b = co0 * self.bernoulli((self.Voltage[i-1]-self.Voltage[i])/Ut) + \
                       co1 * self.bernoulli((self.Voltage[i+1]-self.Voltage[i])/Ut)
                co_c = co1 * self.bernoulli((self.Voltage[i]-self.Voltage[i+1])/Ut)

                a[i], c[i] = co_a, co_c
                if F_scatt:
                    b[i] = -co_b - RecombRate*self.n[i] + (RecombRate**2)*tau_n*(self.p[i]*self.n[i]-ni**2)
                    r[i] = -(co_a*self.p[i-1] - co_b*self.p[i] + co_c*self.p[i+1] + self.G[i] - 
                         RecombRate*(self.p[i]*self.n[i]-ni**2))
                else:
                    b[i] = -co_b 
                    r[i] = -(co_a*self.p[i-1] -co_b*self.p[i] + co_c*self.p[i+1])

            # Dirichlet BC
            # r[1]   += -a[1]*self.p[0];    a[1]   = 0
            # r[-2]  += -c[-2]*self.p[-1];  c[-2]  = 0

            dp = self.tdm_solve(a, b, c, r, n_int)
            diff_sum = 0.0; p_sum = 0.0
            for i in range(1, self.N_points-1):
                self.p[i] += dp[i]
                if self.p[i] < 0: self.p[i] = 0
                diff_sum += abs(dp[i])
                p_sum    += abs(self.p[i])
            if diff_sum <= tol_abs or diff_sum <= tol_rel*p_sum:
                break

        # Electrons
        a.fill(0); b.fill(0); c.fill(0); r.fill(0)
        while True:
            for i in range(1, self.N_points-1):
                RecombRate = 1.0/(tau_n*(self.p[i]) + tau_p*(self.n[i]))
                co0 = Dn[i]/(dx*dx)
                co1 = Dn[i+1]/(dx*dx)
                co_a = co0 * self.bernoulli((self.Voltage[i-1]-self.Voltage[i])/Ut)
                co_b = co1 * self.bernoulli((self.Voltage[i]-self.Voltage[i+1])/Ut) + \
                       co0 * self.bernoulli((self.Voltage[i]-self.Voltage[i-1])/Ut)
                co_c = co1 * self.bernoulli((self.Voltage[i+1]-self.Voltage[i])/Ut)

                a[i], c[i] = co_a, co_c
                if F_scatt:
                    b[i] = -co_b  - RecombRate*self.p[i] + (RecombRate**2)*tau_p*(self.p[i]*self.n[i]-ni**2)
                    r[i] = -(co_a*self.n[i-1] - co_b*self.n[i] + co_c*self.n[i+1] + self.G[i] - 
                         RecombRate*(self.p[i]*self.n[i]-ni**2))
                else:
                    b[i] = -co_b 
                    r[i] = -(co_a*self.n[i-1] - co_b * self.n[i] + co_c*self.n[i+1])

            # Dirichlet BC
            # r[1]   += -a[1]*self.n[0];   a[1]   = 0
            # r[-2]  += -c[-2]*self.n[-1];  c[-2]  = 0

            dn = self.tdm_solve(a, b, c, r, n_int)
            diff_sum = 0.0; n_sum = 0.0
            for i in range(1, self.N_points-1):
                self.n[i] += dn[i]
                if self.n[i] < 0: self.n[i] = 0
                diff_sum += abs(dn[i])
                n_sum    += abs(self.n[i])
            if diff_sum <= tol_abs or diff_sum <= tol_rel*n_sum:
                break

    def solve_ambipolar(self, tol_abs, tol_rel=1e-6):
        """
        Ambipolar continuity equation solver:
        ∂(n_a)/∂t + ∂J_a/∂x = G - R
        단일 carrier 농도 n_a(x), ambipolar diffusion D_a, mobility mu_a 사용.
        Dirichlet BC, generation은 i==1에서만 주입.
        """
        N = self.N_points
        n_int = N - 2
        dx, Ut = self.dx, self.Ut

        # 원래 이동도와 확산계수
        mu_p = np.ones(N)*self.params['mu_p_P']
        mu_n = np.ones(N)*self.params['mu_n_P']
        Dp = mu_p * Ut
        Dn = mu_n * Ut

        # ambipolar 계수
        Da = 2*Dn*Dp/(Dn + Dp + 1e-30)
        mu_a = 2*mu_n*mu_p/(mu_n + mu_p + 1e-30)

        tau_eff = self.params.get('tau_eff', 1e-6)
        G = np.zeros(self.N_points)
        G[0:1] = 2e14*1e6

        # 미지수: n_a (초기엔 equilibrium에서 선택)
        na = 0.5*(self.p + self.n)  # 또는 원하는 초기 guess

        a = np.zeros(N)
        b = np.zeros(N)
        c = np.zeros(N)
        r = np.zeros(N)

        while True:
            a.fill(0); b.fill(0); c.fill(0); r.fill(0)
            for i in range(1, N-1):
                # SG 이산화
                co0 = Da[i]/(dx*dx)
                co1 = Da[i+1]/(dx*dx)
                co_a = co0 * self.bernoulli((self.Voltage[i]   - self.Voltage[i-1]) / Ut)
                co_b = co0 * self.bernoulli((self.Voltage[i-1] - self.Voltage[i])   / Ut) + \
                    co1 * self.bernoulli((self.Voltage[i+1] - self.Voltage[i])   / Ut)
                co_c = co1 * self.bernoulli((self.Voltage[i]   - self.Voltage[i+1]) / Ut)

                # 계수 설정
                a[i], c[i] = co_a, co_c
                b[i] = -co_b
                # recombination linear term
                if tau_eff is not None:
                    b[i] -= 1.0/tau_eff

                # residual: A·na + G_source
                r[i] = co_a*na[i-1] - co_b*na[i] + co_c*na[i+1] - G[i] - na[i]/tau_eff

            # Dirichlet BC for ambipolar carrier at contacts
            # na[0], na[-1] fixed from equilibrium
            r[1]  += -a[1]*na[0];   a[1]  = 0
            r[-2] += -c[-2]*na[-1]; c[-2] = 0

            # Thomas 알고리즘으로 Δna 계산
            delta_na = self.tdm_solve(a, b, c, r, n_int)

            # 갱신 및 수렴 체크
            diff_sum = 0.0
            na_sum   = 0.0
            for i in range(1, N-1):
                na[i] += delta_na[i]
                diff_sum += abs(delta_na[i])
                na_sum   += abs(na[i])

            if diff_sum <= tol_abs or diff_sum <= tol_rel * na_sum:
                break

        # 결과를 기존 p, n 에 분배(optional)
        # 여기서는 양쪽 carrier를 동일하게 나누어 재설정
        self.p = na.copy()
        self.n = na.copy()

    def calculate_current_density(self):
        n_int = self.N_points - 2
        dx, Ut = self.dx, self.Ut
        ec = E_CHARGE
        mu_p = np.ones(self.N_points) * self.params['mu_p_P']
        mu_n = np.ones(self.N_points) * self.params['mu_n_P']
        Dp = np.copy(mu_p) * Ut
        Dn = np.copy(mu_n) * Ut

        Jp = np.zeros(self.N_points)
        Jn = np.zeros(self.N_points)
        Jp_drift = np.zeros(self.N_points)
        Jn_drift = np.zeros(self.N_points)
        Jt = np.zeros(self.N_points)
        JV_average = 0.0
        for i in range(1, self.N_points-1):
            Jp[i] = -ec * Dp[i+1] * (self.bernoulli((self.Voltage[i]-self.Voltage[i+1])/Ut)*self.p[i+1] -
                                      self.bernoulli((self.Voltage[i+1]-self.Voltage[i])/Ut)*self.p[i]) / dx
            Jn[i] = -ec * Dn[i+1] * (self.bernoulli((self.Voltage[i]-self.Voltage[i+1])/Ut)*self.n[i] -
                                      self.bernoulli((self.Voltage[i+1]-self.Voltage[i])/Ut)*self.n[i+1]) / dx
            Jp_drift = ec*mu_p*self.E_field*self.p
            Jn_drift = ec*mu_n*self.E_field*self.n
            Jt[i] = Jp[i] + Jn[i]
            JV_average += Jt[i]
        JV_average /= n_int
        JV_plot = JV_average/10000  # unit conversion
        Jp = Jp/10000  # unit conversion
        Jn = Jn/10000  # unit conversion
        Jt = Jt/10000
        Jp_drift = Jp_drift/10000
        Jn_drift = Jn_drift/10000
        return {'Jp': Jp, 'Jn': Jn, 'Jt': Jt, 'JV_average': JV_average, 'JV_plot': JV_plot, 'Jp_drift':Jp_drift, 'Jn_drift':Jn_drift}

    def run(self, v_applied):
        self.initialize_boundary_conditions(v_applied)
        tol_poisson    = 1e-7
        tol_abs_cont   = tol_poisson
        tol_rel_cont   = 1e-6
        max_iter       = self.params['max_iter']

        V_old = self.Voltage.copy()
        p_old = self.p.copy()
        n_old = self.n.copy()

        for it in range(1, max_iter+1):
            self.solve_poisson(tol=tol_poisson)
            self.solve_continuity(tol_abs=tol_abs_cont, tol_rel=tol_rel_cont)

            dV = np.max(np.abs(self.Voltage - V_old))
            dp = np.max(np.abs(self.p       - p_old))/np.max(np.abs(self.p))
            dn = np.max(np.abs(self.n       - n_old))/np.max(np.abs(self.n))
            print(f"[SCF it={it}] dV={dV:.2e}, dp={dp:.2e}, dn={dn:.2e}")

            if dV<=tol_poisson and dp<=tol_rel_cont and dn<=tol_rel_cont:
                print("SCF converged.")
                break

            V_old[:] = self.Voltage
            p_old[:] = self.p
            n_old[:] = self.n
        else:
            raise RuntimeError("SCF did not converge within max_iter")

        curr = self.calculate_current_density()
        Fp = -self.Voltage - self.Ut * np.log(self.p/self.params['ni'])
        Fn = -self.Voltage + self.Ut * np.log(self.n/self.params['ni'])
        Charge = (self.p - self.n - self.Dop) / 1e6
        E_band_con = -self.Voltage + 0.5*self.eg
        E_band_int = -self.Voltage
        E_band_val = -self.Voltage - 0.5*self.eg
        dV_dx = (self.Voltage[2:self.N_points] - self.Voltage[0:self.N_points-2])/(2*self.dx)
        depletion_length = 0.0
        if dV_dx.size > 0:
            threshold = np.max(np.abs(dV_dx)) * 0.1
            indices = np.where(np.abs(dV_dx) > threshold)[0]
            if indices.size > 0:
                depletion_length = (self.x[indices[-1]+1] - self.x[indices[0]+1]) / 1e-6
        return {
            'v_applied': v_applied,
            'x': self.x/1e-6,
            'Dop_x': self.Dop_x/1e6,
            'Dop_y': self.Dop_y/1e6,
            'Fp': Fp,
            'Fn': Fn,
            'Charge': Charge,
            'E_potential': self.Voltage.copy(),
            'p_plot': self.p/1e6,
            'n_plot': self.n/1e6,
            'E_field_plot': self.E_field/1e2,
            'E_band_con': E_band_con,
            'E_band_int': E_band_int,
            'E_band_val': E_band_val,
            'current_density': curr,
            'depletion_length': depletion_length
        }

def set_default_parameters():
    return {
        'V_min': 0.0,
        'V_max': 0.6,
        'V_step': 0.1,
        'Vd': 0.6,
        'ni': 1.5e10*1e6,
        'semi_length': 20.0e-6,  # 20 µm in meters
        'Na': 1.0e16*1e6,
        'Nd': 1.0e16*1e6,
        'mu_p_P': 200e-4,
        'mu_p_N': 450e-4,
        'mu_n_P': 700e-4,
        'mu_n_N': 1300e-4,
        'tau_n': 0.001e-6,       # in seconds
        'tau_p': 0.001e-6,
        'Na_profile': 'constant',
        'Nd_profile': 'constant',
        'p1': 0.0,
        'n1': 0.0,
        'F_scatt': True,
        'Nx': 100,
        'T': 300
    }

def update_parameters_from_widgets(param_widgets):
    return {
        'V_min': param_widgets[0].value,
        'V_max': param_widgets[1].value,
        'V_step': param_widgets[2].value,
        'Vd': param_widgets[3].value,
        'ni': param_widgets[4].value,
        'semi_length': param_widgets[5].value,
        'Na': param_widgets[6].value,
        'Nd': param_widgets[7].value,
        'mu_p_P': param_widgets[8].value,
        'mu_p_N': param_widgets[9].value,
        'mu_n_P': param_widgets[10].value,
        'mu_n_N': param_widgets[11].value,
        'tau_n': param_widgets[12].value,
        'tau_p': param_widgets[13].value,
        'p1': param_widgets[14].value,
        'n1': param_widgets[15].value,
        'F_scatt': param_widgets[16].value
    }

def run_simulation(params):
    clear_output(wait=True)
    results_list = []
    v_values = np.around(np.arange(params['V_min'], 
                                    params['V_max'] + params['V_step']/2, 
                                    params['V_step']), decimals=1)
    for v in v_values:
        sim = DeviceSimulator(params)
        result = sim.run(v)
        results_list.append(result)
    plot_simulation_results(results_list)

def plot_simulation_results(results_list):
    fig, axes = plt.subplots(3, 3, figsize=(12,10))
    axes = axes.flatten()
    result = results_list[-1]
    x = result['x']
    axes[0].plot(x, result['Dop_x'], label="Dop_x")
    axes[0].plot(x, result['Dop_y'], label="Dop_y")
    axes[0].set_title('Doping Density')
    axes[0].set_xlabel('x (µm)')
    axes[0].set_ylabel('Doping (/cm³)')
    axes[0].legend()

    axes[1].set_title('Carrier Density')
    axes[1].set_xlabel('x (µm)')
    axes[1].set_ylabel('Carrier Density (/cm³)')
    axes[1].set_yscale('log')
    axes[1].plot(x, result['p_plot'], 'r', label='p')
    axes[1].plot(x, result['n_plot'], 'b', label='n')
    axes[1].legend()

    axes[2].plot(x, result['Charge'])
    axes[2].set_title('Net Charge Density')
    axes[2].set_xlabel('x (µm)')
    axes[2].set_ylabel('Charge (/cm³)')

    axes[3].plot(x, result['E_field_plot'])
    axes[3].set_title('Electric Field')
    axes[3].set_xlabel('x (µm)')
    axes[3].set_ylabel('E Field (kV/cm)')

    axes[4].plot(x, result['E_potential'])
    axes[4].set_title('Electrostatic Potential')
    axes[4].set_xlabel('x (µm)')
    axes[4].set_ylabel('Potential (V)')

    shift = result['v_applied'] / 2
    axes[5].plot(x, result['E_band_con'] + shift, label='Conduction Band')
    axes[5].plot(x, result['E_band_int'] + shift, '--', label='Intrinsic Level')
    axes[5].plot(x, result['E_band_val'] + shift, label='Valence Band')
    axes[5].plot(x, result['Fp'] + shift, label='Fp')
    axes[5].plot(x, result['Fn'] + shift, label='Fn')
    axes[5].set_title('Energy Bands')
    axes[5].set_xlabel('x (µm)')
    axes[5].set_ylabel('Energy (eV)')
    axes[5].legend()

    v_vals = [r['v_applied'] for r in results_list]
    JV_vals = [r['current_density']['JV_plot'] for r in results_list]
    axes[6].plot(v_vals, JV_vals, 'bo-')
    axes[6].set_title('J-V Curve')
    axes[6].set_xlabel('Bias (V)')
    axes[6].set_ylabel('Current Density (A/cm²)')

    axes[7].set_yscale('log')
    axes[7].plot(x, result['current_density']['Jt'], label='J_total')
    axes[7].set_title('Current Density Distribution')
    axes[7].set_xlabel('x (µm)')
    axes[7].set_ylabel('Current Density (A/cm²)')
    axes[7].legend()

    depletion_vals = [r['depletion_length'] for r in results_list]
    axes[8].plot(v_vals, depletion_vals, 'gs-')
    axes[8].set_title('Depletion Region Length')
    axes[8].set_xlabel('Bias (V)')
    axes[8].set_ylabel('Depletion Length (µm)')

    plt.tight_layout()
    plt.show()

def main():
    default_params = set_default_parameters()
    run_simulation(default_params)

if __name__ == '__main__':
    main()


