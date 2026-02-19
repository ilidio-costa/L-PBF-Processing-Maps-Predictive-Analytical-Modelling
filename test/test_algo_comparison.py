import time
import numpy as np
import sys
import os
from scipy.integrate import quad
from scipy.optimize import newton
import logging

# --- FIX: Add Project Root to Path ---
# This allows importing from 'src' even if running from 'test/'
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# --- Setup Logger ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# --- Import Existing Functions ---
try:
    from src.physics import get_eagar_tsai_dimensions, get_melt_pool_dimensions
except ImportError:
    logger.error(f"Could not import src.physics. Check that '{project_root}' contains a 'src' folder.")
    raise

# --- Improved Annex C Function (Robust Integration) ---
def get_melt_pool_dimensions_analytical(P, v, a, material, T_0=298.15):
    """
    Robust Implementation of Annex C: Gradient-Based Newton-Raphson Solver.
    """
    rho = material['rho']
    Cp = material['C_p']
    k = material['k']
    A = material['A']
    T_m = material['T_m']
    alpha = material['alpha']

    # Pre-factor C (Eq 34)
    C = (A * P / (np.pi * k)) * np.sqrt(alpha / np.pi)

    # --- Robust Integration Logic ---
    # Calculate a safe upper limit for integration (when T decays to ~0)
    limit_s = (a**2 / alpha) * 10.0
    
    def robust_quad(func):
        # Start at 1e-9 to avoid singularity at s=0
        val, _ = quad(func, 1e-9, limit_s, points=[a/v], limit=100)
        return val

    # --- Kernel Functions (Eq 35 & Derivatives) ---
    def _integrand_common(s, xi, y, z):
        denom_xy = 4 * alpha * s + a**2
        denom_z = 4 * alpha * s
        
        exp_z = -(z**2) / denom_z
        exp_xy = -((y**2 + (xi + v*s)**2) / denom_xy)
        
        return (1.0 / (denom_xy * np.sqrt(s))) * np.exp(exp_z + exp_xy)

    def temperature(xi, y, z): # Eq 33
        return C * robust_quad(lambda s: _integrand_common(s, xi, y, z)) + T_0

    def dT_dxi(xi, y, z): # Eq 38
        def integrand(s):
            phi = _integrand_common(s, xi, y, z)
            term = -2 * (xi + v*s) / (4 * alpha * s + a**2)
            return phi * term
        return C * robust_quad(integrand)

    def dT_dy(xi, y, z): # Eq 40
        def integrand(s):
            phi = _integrand_common(s, xi, y, z)
            term = -2 * y / (4 * alpha * s + a**2)
            return phi * term
        return C * robust_quad(integrand)

    def dT_dz(xi, y, z): # Eq 42
        def integrand(s):
            phi = _integrand_common(s, xi, y, z)
            term = -2 * z / (4 * alpha * s)
            return phi * term
        return C * robust_quad(integrand)

    # --- Newton-Raphson Solver ---
    try:
        # 1. Peak Location (Eq 43)
        xi_peak = newton(lambda x: dT_dxi(x, 0, 0), x0=-a/2.0, tol=1e-6)
        
        if temperature(xi_peak, 0, 0) < T_m:
            return 0.0, 0.0, 0.0, 0.0, 0.0

        # 2. Width (Eq 44)
        y_star = newton(lambda y: temperature(xi_peak, y, 0) - T_m, x0=a, 
                        fprime=lambda y: dT_dy(xi_peak, y, 0), tol=1e-6)
        width = 2 * abs(y_star)

        # 3. Depth (Eq 45)
        z_star = newton(lambda z: temperature(xi_peak, 0, z) - T_m, x0=a/2.0, 
                        fprime=lambda z: dT_dz(xi_peak, 0, z), tol=1e-6)
        depth = abs(z_star)

        # 4. Length (Eq 46)
        xi_front = newton(lambda x: temperature(x, 0, 0) - T_m, x0=xi_peak + a, 
                          fprime=lambda x: dT_dxi(x, 0, 0), tol=1e-6)
        xi_back = newton(lambda x: temperature(x, 0, 0) - T_m, x0=xi_peak - a*2, 
                         fprime=lambda x: dT_dxi(x, 0, 0), tol=1e-6)
        length = xi_front - xi_back

        return length, width, depth, xi_back, xi_front

    except RuntimeError:
        return 0.0, 0.0, 0.0, 0.0, 0.0

# --- Benchmark Logic ---
def run_benchmark():
    # Mock Material (NiTi)
    material = {
        "name": "NiTi", "rho": 6450.0, "C_p": 800.0, "k": 18.0, 
        "T_m": 1583.0, "A": 0.32, "alpha": 3.48e-6
    }
    
    P, v, a = 200.0, 1.0, 50e-6

    logger.info("="*85)
    logger.info(f"BENCHMARK V3: P={P}W, v={v}m/s, a={a*1e6}um | Material: {material['name']}")
    logger.info("="*85)
    
    # 1. Global Search (Baseline)
    t0 = time.time()
    res_global = get_eagar_tsai_dimensions(P, v, a, material, resolution=150)
    t_global = time.time() - t0
    
    # 2. 1D Heuristic (Current)
    t0 = time.time()
    res_heuristic = get_melt_pool_dimensions(P, v, a, material)
    t_heuristic = time.time() - t0
    
    # 3. Newton-Raphson (Improved)
    t0 = time.time()
    res_newton = get_melt_pool_dimensions_analytical(P, v, a, material, T_0=298.15)
    t_newton = time.time() - t0

    # Output
    headers = f"{'METHOD':<20} | {'LENGTH (um)':<12} | {'WIDTH (um)':<12} | {'DEPTH (um)':<12} | {'TIME (ms)':<10} | {'SPEEDUP':<8}"
    logger.info(headers)
    logger.info("-" * 85)
    
    def print_row(name, res, t, t_base):
        l, w, d = res[0]*1e6, res[1]*1e6, res[2]*1e6
        speedup = t_base / t if t > 0 else 0
        logger.info(f"{name:<20} | {l:>12.2f} | {w:>12.2f} | {d:>12.2f} | {t*1000:>10.2f} | {speedup:>7.1f}x")

    print_row("Global Search", res_global, t_global, t_global)
    print_row("1D Heuristic", res_heuristic, t_heuristic, t_global)
    print_row("Newton-Raphson", res_newton, t_newton, t_global)

    # Check Accuracy
    err_newt = abs(res_newton[0] - res_global[0]) / (res_global[0] or 1) * 100
    if err_newt < 1.0 and res_newton[0] > 0:
        logger.info("\n[SUCCESS] Newton-Raphson works and matches Global Search!")
    else:
        logger.warning(f"\n[FAIL] Newton-Raphson still failing or deviant (Error: {err_newt:.2f}%)")



if __name__ == "__main__":
    run_benchmark()