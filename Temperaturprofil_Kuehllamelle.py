"""
PROJECT: Steady-State 1D Cooling Fin Simulation
DESCRIPTION:
This script computes the temperature distribution along a rectangular cooling fin
using the Finite Difference Method (FDM). It validates the numerical results
against an analytical solution and performs a grid sensitivity analysis.

METHODOLOGY:
- Numerical: FDM with 2nd order central differences.
- Boundary Conditions: Dirichlet at base (x=0) and comparison for tip (x=L) between
  adiabatic (Q=0) and Robin (Q_conv = Q_cond).
- Solver: Thomas Algorithm via scipy.linalg.solve_banded
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_banded


def solve_temperature_fin(n_points, length, width, thickness, lam, h_conv, T_base, T_inf, tip_bc="adiabatic"):
    """
    Compute the 1D steady-state temperature distribution along a rectangular fin.

    Parameters
    ----------
    n_points  : int   – number of grid points
    length    : float – fin length [m]
    width     : float – fin width [m]
    thickness : float – fin thickness [m]
    lam       : float – thermal conductivity [W/(m·K)]
    h_conv    : float – convective heat transfer coefficient [W/(m²·K)]
    T_base    : float – base temperature [K]
    T_inf     : float – ambient temperature [K]
    tip_bc    : str   – tip boundary condition: 'adiabatic' or 'convective'

    Returns
    -------
    x     : ndarray – position vector [m]
    T_num : ndarray – numerical temperature distribution [K]
    T_ana : ndarray – analytical temperature distribution [K]
    """
    if tip_bc not in ("adiabatic", "convective"):
        raise ValueError(f"tip_bc must be 'adiabatic' or 'convective', got '{tip_bc}'")

    A_cross_section = width * thickness
    perimeter = 2 * (width + thickness)
    dx = length / (n_points - 1)
    x = np.linspace(0, length, n_points)
    theta_b = T_base - T_inf
    m2 = (h_conv * perimeter) / (lam * A_cross_section)
    m = np.sqrt(m2)

    # Analytical solution (VDI Heat Atlas)
    if tip_bc == "adiabatic":
        T_ana = T_inf + theta_b * (np.cosh(m * (length - x)) / np.cosh(m * length))
    else:
        h_m_lam = h_conv / (m * lam)
        numerator = np.cosh(m * (length - x)) + h_m_lam * np.sinh(m * (length - x))
        denominator = np.cosh(m * length) + h_m_lam * np.sinh(m * length)
        T_ana = T_inf + theta_b * (numerator / denominator)

    # Numerical solution: tridiagonal system A * T = b
    A_matrix = np.zeros((3, n_points))
    b_vec = np.zeros(n_points)
    coeff = m2 * dx**2

    A_matrix[0, 1:] = 1.0            # upper diagonal
    A_matrix[1, 1:-1] = -(2 + coeff) # main diagonal
    A_matrix[2, :-1] = 1.0           # lower diagonal

    # BC at x=0: Dirichlet
    A_matrix[1, 0] = 1.0
    A_matrix[0, 1] = 0.0
    b_vec[0] = T_base

    if tip_bc == "adiabatic":
        # Ghost-point symmetry: dT/dx = 0 → T_{N+1} = T_{N-1}
        A_matrix[1, -1] = -(2 + coeff)
        A_matrix[2, -2] = 2.0
        b_vec[-1] = -coeff * T_inf
    else:
        # Convective tip: Robin BC, 2nd-order accurate
        alpha_ratio = (h_conv * dx) / lam
        A_matrix[1, -1] = -(2 + coeff + 2 * alpha_ratio)
        A_matrix[2, -2] = 2.0
        b_vec[-1] = -T_inf * (coeff + 2 * alpha_ratio)

    b_vec[1:-1] = -coeff * T_inf

    T_num = solve_banded((1, 1), A_matrix, b_vec)
    return x, T_num, T_ana


def calculate_h_conv(v_flow, L_char, T_film):
    """
    Convective coefficient for forced flow over a flat plate (laminar: Re < 5e5).
    Flow approaches the fin from its thin edge and travels along the fin length.
    Air properties evaluated at the film temperature T_film = (T_base + T_inf) / 2.

    Parameters
    ----------
    v_flow : float – free-stream air velocity [m/s]
    L_char : float – characteristic length in flow direction [m]
    T_film : float – film temperature [K]

    Returns
    -------
    h  : float – convective heat transfer coefficient [W/(m²·K)]
    Re : float – Reynolds number
    Nu : float – Nusselt number
    """
    # Dynamic viscosity via Sutherland's law
    mu_ref, T_ref, S = 1.716e-5, 273.15, 110.4
    mu = mu_ref * (T_film / T_ref) ** 1.5 * (T_ref + S) / (T_film + S)

    rho   = 101325 / (287.058 * T_film)          # ideal gas
    k_air = 0.0241 * (T_film / 273.15) ** 0.82   # power-law fit for air
    Pr    = 0.707                                  # approx. constant for air
    nu    = mu / rho

    Re = v_flow * L_char / nu
    if Re < 5e5:
        Nu = 0.664 * Re ** 0.5 * Pr ** (1 / 3)
    else:
        Nu = (0.037 * Re ** 0.8 - 871) * Pr ** (1 / 3)   # mixed BL

    h = Nu * k_air / L_char
    return h, Re, Nu


def plot_fin_comparison(x, T_num_ad, T_ana_ad, T_num_conv, T_ana_conv, save_path=None):
    plt.figure(figsize=(10, 6))
    x_cm = x * 100

    plt.plot(x_cm, T_num_ad - 273.15,   color='#1f77b4', linewidth=2.5, label='Numerical: Adiabatic Tip')
    plt.plot(x_cm, T_ana_ad - 273.15,   color='black',   linestyle='--', linewidth=1, label='Analytical: Adiabatic Tip')
    plt.plot(x_cm, T_num_conv - 273.15, color='#d62728', linewidth=2.5, label='Numerical: Convective Tip')
    plt.plot(x_cm, T_ana_conv - 273.15, color='black',   linestyle=':', linewidth=1, label='Analytical: Convective Tip')

    all_T = np.concatenate([T_num_ad, T_ana_ad, T_num_conv, T_ana_conv]) - 273.15
    margin = 0.5
    plt.xlim(0, x_cm[-1])
    plt.ylim(all_T.min() - margin, all_T.max() + margin)
    plt.title('Comparison of the Influence of Different Boundary Conditions at the Fin Tip', fontsize=13)
    plt.xlabel('Position $x$ [cm]', fontsize=11)
    plt.ylabel('Temperature $T$ [°C]', fontsize=11)
    plt.legend(loc='best', frameon=True)
    plt.grid(True, which='both', linestyle=':', alpha=0.5)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_grid_sensitivity(params, tip_bc="adiabatic", save_path=None):
    """Convergence study: max absolute error vs. grid spacing (expects 2nd-order)."""
    n_values = [5, 10, 20, 50, 100, 200, 500, 1000]
    dx_values = []
    max_errors = []

    for n in n_values:
        x, T_num, T_ana = solve_temperature_fin(n, **params, tip_bc=tip_bc)
        dx_values.append(params['length'] / (n - 1))
        max_errors.append(np.max(np.abs(T_num - T_ana)))

    dx_values = np.array(dx_values)
    max_errors = np.array(max_errors)

    # 2nd-order reference line anchored at the coarsest grid point
    ref_x = np.array([dx_values[0], dx_values[-1]])
    ref_y = max_errors[0] * (ref_x / dx_values[0]) ** 2

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(dx_values, max_errors, 'o-', color='#1f77b4', linewidth=2, markersize=6, label='Max. absolute error')
    ax.loglog(ref_x, ref_y, 'k--', linewidth=1, label='2nd-order reference slope')
    ax.set_xlabel('Grid spacing $\\Delta x$ [m]', fontsize=11)
    ax.set_ylabel('Max. absolute error [K]', fontsize=11)
    ax.set_title(f'Grid Sensitivity Analysis ({tip_bc.capitalize()} Tip BC)', fontsize=13)
    ax.legend()
    ax.grid(True, which='both', linestyle=':', alpha=0.5)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    # Geometry and material properties
    length, width, thickness = 0.15, 0.12, 0.003
    lam    = 200.0    # thermal conductivity aluminium [W/(m·K)]
    T_base = 353.15   # base temperature [K]  (80 °C)
    T_inf  = 300.0    # ambient temperature [K] (27 °C)

    # Convective coefficient from flat-plate correlation
    # Flow: air at 2 m/s, approaches from thin edge → L_char = fin length
    v_flow = 2.0
    T_film = (T_base + T_inf) / 2
    h_conv, Re, Nu = calculate_h_conv(v_flow, length, T_film)
    print(f"Film temperature:              {T_film - 273.15:.1f} °C")
    print(f"Reynolds number:               {Re:.0f}")
    print(f"Nusselt number:                {Nu:.2f}")
    print(f"Convective coefficient h_conv: {h_conv:.2f} W/(m²·K)")
    print()

    params = {
        'length': length, 'width': width, 'thickness': thickness,
        'lam': lam, 'h_conv': h_conv, 'T_inf': T_inf, 'T_base': T_base
    }

    x, T_num_ad, T_ana_ad    = solve_temperature_fin(50, **params, tip_bc='adiabatic')
    _, T_num_conv, T_ana_conv = solve_temperature_fin(50, **params, tip_bc='convective')

    diff_bc_num       = np.abs(T_num_ad[-1]   - T_num_conv[-1])
    diff_bc_ana       = np.abs(T_ana_ad[-1]   - T_ana_conv[-1])
    diff_num_ana_ad   = np.abs(T_num_ad[-1]   - T_ana_ad[-1])
    diff_num_ana_conv = np.abs(T_num_conv[-1] - T_ana_conv[-1])

    print(f"Fin tip temperature difference between adiabatic and Robin BC: {diff_bc_num:.4f} K (numerical)")
    print(f"Fin tip temperature difference between adiabatic and Robin BC: {diff_bc_ana:.4f} K (analytical)")
    print(f"Fin tip error (numerical vs. analytical): {diff_num_ana_ad:.5f} K  (adiabatic)")
    print(f"Fin tip error (numerical vs. analytical): {diff_num_ana_conv:.5f} K  (Robin)")

    plot_fin_comparison(x, T_num_ad, T_ana_ad, T_num_conv, T_ana_conv,
                        save_path='results/temperature_profile.png')
    plot_grid_sensitivity(params, tip_bc='adiabatic',
                          save_path='results/grid_sensitivity.png')
