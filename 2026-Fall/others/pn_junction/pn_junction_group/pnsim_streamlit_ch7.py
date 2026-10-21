import json
import numpy as np
import scipy.constants as const
from pnsim import DeviceSimulator
from visual import PlotFactory
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.constants as const

# Constants
EPSILON_0 = const.epsilon_0       # F/m
K_B       = const.k               # J/K
Pi        = const.pi
E_CHARGE  = const.e               # C
HP        = const.h               # J s\
m_e       = const.m_e

# -----------------------------------------------------------------------------
# Default Simulation Parameters
# -----------------------------------------------------------------------------
def set_default_parameters():
    return {
        'V_min': 0.0,
        'V_max': 0.2,
        'V_step': 0.1,
        'ni': 1.5e10,
        'semi_length': 20.0,
        'Na': 1.0e17,
        'Nd': 1.0e16,
        'mu_p_P': 200.0,
        'mu_n_P': 700.0,
        'tau_n': 0.001,
        'tau_p': 0.001,
        'F_scatt': True,
        'g_type': 'uniform',
        'g_density': 1e15,
        'Nx': 1000,
        'Temp': 300.0,
        'max_iter': 300
    }

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
def serialize_params(params):
    p = params.copy()
    p['ni'] *= 1e6
    p['semi_length'] *= 1e-6
    p['mu_p_P'] *= 1e-4
    p['mu_n_P'] *= 1e-4
    p['tau_n'] *= 1e-6
    p['tau_p'] *= 1e-6
    if 'g_density' in p:
        p['g_density'] *= 1e6
    p['Na_profile'] = np.array(p['Na_profile'])
    p['Nd_profile'] = np.array(p['Nd_profile'])
    return p


def compute_profile(profile, x, **kwargs):
    if profile == 'constant':
        return np.full_like(x, kwargs.get('c', 0))
    elif profile == 'linear':
        L = kwargs.get('L')
        return kwargs.get('a') * (L - x) / L + kwargs.get('b') * x / L
    else:
        return kwargs.get('a') + kwargs.get('b') * np.exp(-x / (kwargs.get('d') * 1e4))

@st.cache_data
def run_and_cache(params_json: str, params: dict):
    sim_params = serialize_params(params)
    sim_params.update({'p1': 0.0, 'n1': 0.0})
    if params.get('F_scatt'):
        sim_params['gen_type'] = params['g_type']
        sim_params['gen_density'] = params['g_density']
    v_vals = np.around(
        np.arange(params['V_min'], params['V_max'] + params['V_step'] / 2, params['V_step']), 6
    )
    results = []
    for v in v_vals:
        sim = DeviceSimulator(sim_params)
        results.append(sim.run(v))
    return v_vals, results

# -----------------------------------------------------------------------------
# App Layout
# -----------------------------------------------------------------------------
st.set_page_config(layout='wide')
params_default = set_default_parameters()

# Sidebar: Simulation Parameters
st.sidebar.header('Simulation Parameters')
V_min = st.sidebar.number_input('Minimum Voltage (V)', value=params_default['V_min'], format='%.4f')
V_max = st.sidebar.number_input('Maximum Voltage (V)', value=params_default['V_max'], format='%.4f')
V_step = st.sidebar.number_input('Voltage Step (V)', value=params_default['V_step'], format='%.4f')
mn = st.sidebar.number_input('electron effective mass (cm⁻³)', value=1.08, format='%.4e')
mp = st.sidebar.number_input('hole effective mass (cm⁻³)', value=0.56, format='%.4e')
eg = st.sidebar.number_input('band gap (eV)', value=1.12, format='%.4e')
Temp = st.sidebar.number_input('Operating Temperature (K)', value=params_default['Temp'], format='%.4f')

# get ni
Ut = K_B * Temp / E_CHARGE
Nc = 2*((2*Pi*mn*m_e*K_B * Temp)/(HP**2))**(3/2)
Nv = 2*((2*Pi*mp*m_e*K_B * Temp)/(HP**2))**(3/2)
ni = np.sqrt(Nc*Nv*np.exp(-eg/Ut))*1e-6
st.sidebar.text(f'ni = {ni:.4e}')
Nx = st.sidebar.number_input('Grid Resolution (Nx)', value=params_default['Nx'], step=1)
max_iter = st.sidebar.number_input('Max Iteration', value=params_default['max_iter'], step=10)
mu_n_P = st.sidebar.number_input('Electron Mobility (cm²/Vs)', value=params_default['mu_n_P'], format="%.4f")
mu_p_P = st.sidebar.number_input('Hole Mobility (cm²/Vs)', value=params_default['mu_p_P'], format="%.4f")
tau_n = st.sidebar.number_input('Electron Lifetime (µs)', value=params_default['tau_n'], format="%.4f")
tau_p = st.sidebar.number_input('Hole Lifetime (µs)', value=params_default['tau_p'], format="%.4f")

# Sidebar: Semiconductor Segments
st.sidebar.header('Semiconductor Segments')
num_segments = st.sidebar.number_input('Number of Segments', min_value=1, max_value=3, value=1, step=1)
segments = []
total_length = 0.0
for i in range(num_segments):
    exp = st.sidebar.expander(f'Segment {i+1}')
    L = exp.number_input(f'Length of Segment {i+1} (µm)', value=params_default['semi_length'], format='%.4f', key=f'L_{i}')
    total_length += L
    # Na Profile
    Na_type = exp.selectbox('Na Profile Type', ['constant', 'linear', 'exponential'], index=0, key=f'Na_t_{i}')
    Na_params = {}
    if Na_type == 'constant':
        Na_params['c'] = exp.number_input('Na: Constant c (cm⁻³)', value=params_default['Na'], format='%.4e', key=f'Na_c_{i}')
    elif Na_type == 'linear':
        Na_params['a'] = exp.number_input('Na: Linear left (a)', value=0.0, format='%.4e', key=f'Na_a_{i}')
        Na_params['b'] = exp.number_input('Na: Linear right (b)', value=params_default['Na'], format='%.4e', key=f'Na_b_{i}')
    else:
        Na_params['a'] = exp.number_input('Na: Exp a', value=0.0, format='%.4e', key=f'Na_ea_{i}')
        Na_params['b'] = exp.number_input('Na: Exp b', value=params_default['Na'], format='%.4e', key=f'Na_eb_{i}')
        Na_params['d'] = exp.number_input('Na: Exp d (µm)', value=params_default['semi_length'], format='%.4f', key=f'Na_d_{i}')
    # Nd Profile
    Nd_type = exp.selectbox('Nd Profile Type', ['constant', 'linear', 'exponential'], index=0, key=f'Nd_t_{i}')
    Nd_params = {}
    if Nd_type == 'constant':
        Nd_params['c'] = exp.number_input('Nd: Constant c (cm⁻³)', value=params_default['Nd'], format='%.4e', key=f'Nd_c_{i}')
    elif Nd_type == 'linear':
        Nd_params['a'] = exp.number_input('Nd: Linear left (a)', value=0.0, format='%.4e', key=f'Nd_a_{i}')
        Nd_params['b'] = exp.number_input('Nd: Linear right (b)', value=params_default['Nd'], format='%.4e', key=f'Nd_b_{i}')
    else:
        Nd_params['a'] = exp.number_input('Nd: Exp a', value=0.0, format='%.4e', key=f'Nd_ea_{i}')
        Nd_params['b'] = exp.number_input('Nd: Exp b', value=params_default['Nd'], format='%.4e', key=f'Nd_eb_{i}')
        Nd_params['d'] = exp.number_input('Nd: Exp d (µm)', value=params_default['semi_length'], format='%.4f', key=f'Nd_d_{i}')
    segments.append({'L': L, 'Na_type': Na_type, 'Na_params': Na_params,
                     'Nd_type': Nd_type, 'Nd_params': Nd_params})

# Generation & Recombination
F_scatt = st.sidebar.checkbox('Consider Generations/Recombinations', value=params_default['F_scatt'])
if F_scatt:
    gen_type = st.sidebar.radio('Generation Type', ['uniform', 'end'], index=0)
    gen_density = st.sidebar.number_input('Excess carrier concentration (cm⁻³)', value=params_default['g_density'], format='%.4e')
else:
    gen_type, gen_density = params_default['g_type'], params_default['g_density']

# Build combined profiles with fixed total Nx (ensure Nx+2 points)
x_full = np.linspace(0, total_length, Nx+2)
na_full = np.zeros_like(x_full)
nd_full = np.zeros_like(x_full)
# Compute segment boundaries
boundaries = np.cumsum([0] + [seg['L'] for seg in segments])
for idx, seg in enumerate(segments):
    start, end = boundaries[idx], boundaries[idx+1]
    if idx == len(segments)-1:
        mask = (x_full >= start) & (x_full <= end)
    else:
        mask = (x_full >= start) & (x_full < end)
    x_local = x_full[mask] - start
    na_full[mask] = compute_profile(seg['Na_type'], x_local, **seg['Na_params'], L=seg['L'])
    nd_full[mask] = compute_profile(seg['Nd_type'], x_local, **seg['Nd_params'], L=seg['L'])

# Plot Initial Doping Profile
fig = go.Figure()
fig.add_trace(go.Scatter(x=x_full, y=na_full, mode='lines', name='Na(x)'))
fig.add_trace(go.Scatter(x=x_full, y=nd_full, mode='lines', name='Nd(x)'))
fig.update_layout(xaxis_title='Position (µm)', yaxis_title='Doping (cm⁻³)', height=500)

st.plotly_chart(fig, width="stretch")

# Start Simulation
if st.sidebar.button('Start Simulation'):
    params = {
        'V_min': V_min, 'V_max': V_max, 'V_step': V_step,
        'ni': ni, 'semi_length': total_length,
        'mu_p_P': mu_p_P, 'mu_n_P': mu_n_P,
        'tau_n': tau_n, 'tau_p': tau_p,
        'F_scatt': F_scatt, 'Nx': Nx, 'Temp': Temp, 'max_iter': max_iter,
        'Na_profile': (na_full * 1e6).tolist(), 'Nd_profile': (nd_full * 1e6).tolist(),
        'g_type': gen_type, 'g_density': gen_density, 'Eg' : eg 
    }
    key = json.dumps(params, sort_keys=True)
    v_vals, results = run_and_cache(key, params)
    st.session_state['cache'] = {'v': v_vals, 'res': results}

# Display Simulation Results
if 'cache' in st.session_state:
    st.header('Simulation Result')
    cache = st.session_state['cache']
    v_vals, results = cache['v'], cache['res']
    if 'show_all' not in st.session_state:
        st.session_state['show_all'] = False
    st.sidebar.header('Simulation Result')
    vd = st.sidebar.select_slider('Bias Voltage (V)', options=[float(v) for v in v_vals], value=float(v_vals.min()))
    fig_type = st.sidebar.selectbox('Figure:', ['J-V Curve', 'Depletion Region Length', 'Doping Density', 'Carrier Density', 'Net Charge Density', 'Electric Field', 'Electrostatic Potential', 'Energy Bands', 'Current Density Distribution'], index=0)
    if not st.session_state['show_all'] and st.sidebar.button('Show All Figures'):
        st.session_state['show_all'] = True
    elif st.session_state['show_all'] and st.sidebar.button('Go Back'):
        st.session_state['show_all'] = False
    jv_vals = [res['current_density']['JV_plot'] for res in results]
    dep_vals = [res['depletion_length'] for res in results]
    if st.session_state['show_all']:
        types = ['J-V Curve', 'Depletion Region Length', 'Doping Density', 'Carrier Density', 'Net Charge Density', 'Electric Field', 'Electrostatic Potential', 'Energy Bands', 'Current Density Distribution']
        all_fig = make_subplots(rows=3, cols=3, subplot_titles=types)
        for i, t in enumerate(types):
            row, col = divmod(i, 3)
            if t == 'J-V Curve':
                fig_i = go.Figure()
                fig_i.add_trace(go.Scatter(x=v_vals, y=jv_vals, mode='lines', name='J-V'))
                idx = int(np.argmin(np.abs(v_vals - vd)))
                fig_i.add_trace(go.Scatter(x=[vd], y=[jv_vals[idx]], mode='markers', marker=dict(size=12)))
            elif t == 'Depletion Region Length':
                fig_i = go.Figure()
                fig_i.add_trace(go.Scatter(x=v_vals, y=dep_vals, mode='lines', name='Dep Len'))
                idx = int(np.argmin(np.abs(v_vals - vd)))
                fig_i.add_trace(go.Scatter(x=[vd], y=[dep_vals[idx]], mode='markers', marker=dict(size=12)))
            else:
                inst = PlotFactory.create_plot(t)
                idx = int(np.argmin(np.abs(v_vals - vd)))
                fig_i = inst.build(results[idx])
            for tr in fig_i.data:
                all_fig.add_trace(tr, row=row+1, col=col+1)
        all_fig.update_layout(height=900, width=1200)
        st.plotly_chart(all_fig, width="stretch")
    else:
        if fig_type == 'J-V Curve':
            main_plot = go.Figure()
            main_plot.add_trace(go.Scatter(x=v_vals, y=jv_vals, mode='lines', name='J-V'))
            idx = int(np.argmin(np.abs(v_vals - vd)))
            main_plot.add_trace(go.Scatter(x=[vd], y=[jv_vals[idx]], mode='markers', marker=dict(size=12)))
            # main_plot.update_yaxes(type='log')
        elif fig_type == 'Depletion Region Length':
            main_plot = go.Figure()
            main_plot.add_trace(go.Scatter(x=v_vals, y=dep_vals, mode='lines', name='Dep Len'))
            idx = int(np.argmin(np.abs(v_vals - vd)))
            main_plot.add_trace(go.Scatter(x=[vd], y=[dep_vals[idx]], mode='markers', marker=dict(size=12)))
        else:
            inst = PlotFactory.create_plot(fig_type)
            idx = int(np.argmin(np.abs(v_vals - vd)))
            main_plot = inst.build(results[idx])
        main_plot.update_layout(title=f"{fig_type} at V={vd} V", height=600, width=800)
        st.plotly_chart(main_plot, width="stretch")