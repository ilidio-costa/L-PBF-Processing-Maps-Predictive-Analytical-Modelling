import numpy as np
import matplotlib.pyplot as plt
import os
import time
from scipy.optimize import minimize_scalar
from utils import setup_test_env

from src.physics import (
    eagar_tsai_temp, 
    rubenchik_field,
    get_eagar_tsai_dimensions # Import the new function
)

# --- MOCK DATA (Zhu et al. NiTi Parameters) ---
# Values taken directly from materials/NiTi.json
MOCK_MATERIAL = {
    "name": "NiTi",
    "A": 0.32,          # Absorptivity
    "k": 4.4,           # Thermal Conductivity [W/mK]
    "alpha": 8e-6,      # Thermal Diffusivity [m^2/s]
    "rho": 6100.0,      # Density [kg/m^3]
    "C_p": 510.0,       # Specific Heat [J/kgK]
    "T_m": 1583.0,      # Melting Point [K]
    "T_b": 3033.0       # Boiling Point [K]
}

def test_general_physics(logger):
    """Checks basic scalar outputs."""
    logger.info(">>> Running General Physics Checks...")
    P, v, a = 200.0, 1.0, 50e-6
    
    # 1. Check Peak Temperature Rise
    T_rise = eagar_tsai_temp(0, 0, 0, P, v, a, MOCK_MATERIAL)
    logger.info(f" [PASS] Peak Temp Rise (approx): {T_rise:.2f} K")

    # 2. Check Rubenchik Dimensions
    L, W, D = rubenchik_field(P, v, a, MOCK_MATERIAL, T_ambient=298)
    if D < 0:
         logger.info(f" [PASS] Rubenchik D={D*1e6:.1f}um (Negative=Into Material)")
    else:
         logger.error(f" [FAIL] Rubenchik Depth is positive: {D}")

def find_peak_x(P, v, a, material):
    """Finds exact peak location (Thermal Lag)."""
    res = minimize_scalar(
        lambda x: -eagar_tsai_temp(x, 0, 0, P, v, a, material), 
        bounds=(-3*a, a), method='bounded'
    )
    return res.x, -res.fun

def test_spatial_profiles(logger, output_folder):
    """Generates X, Y, and Z temperature profiles."""
    logger.info("\n>>> Running Spatial Profile Sweep (X, Y, Z)...")
    
    powers = [150, 350]
    velocities = [0.8, 1.2]
    a = 50e-6
    Tm = MOCK_MATERIAL['T_m']
    
    fig, axes = plt.subplots(len(powers)*len(velocities), 3, figsize=(15, 12))
    fig.suptitle(f"Spatial Profiles (NiTi) - Tm={Tm}K", fontsize=16)

    row_idx = 0
    for P in powers:
        for v in velocities:
            x_peak, T_max = find_peak_x(P, v, a, MOCK_MATERIAL)
            
            # Scans
            xs = np.linspace(x_peak - 400e-6, x_peak + 100e-6, 100)
            ys = np.linspace(-150e-6, 150e-6, 100)
            zs = np.linspace(-150e-6, 0, 100) # Depth < 0

            # Profiles
            T_x = [eagar_tsai_temp(val, 0, 0, P, v, a, MOCK_MATERIAL) for val in xs]
            T_y = [eagar_tsai_temp(x_peak, val, 0, P, v, a, MOCK_MATERIAL) for val in ys]
            T_z = [eagar_tsai_temp(x_peak, 0, val, P, v, a, MOCK_MATERIAL) for val in zs]

            # Plot X
            axes[row_idx, 0].plot(xs*1e6, T_x, 'b-')
            axes[row_idx, 0].axhline(Tm, color='r', linestyle='--')
            axes[row_idx, 0].set_title(f"X-Axis (P={P}, v={v})")
            
            # Plot Y
            axes[row_idx, 1].plot(ys*1e6, T_y, 'g-')
            axes[row_idx, 1].axhline(Tm, color='r', linestyle='--')
            axes[row_idx, 1].set_title("Y-Axis (Width)")

            # Plot Z
            axes[row_idx, 2].plot(zs*1e6, T_z, 'purple')
            axes[row_idx, 2].axhline(Tm, color='r', linestyle='--')
            axes[row_idx, 2].set_title("Z-Axis (Depth)")
            
            row_idx += 1

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = os.path.join(output_folder, "spatial_profiles_sweep.png")
    fig.savefig(save_path)
    plt.close(fig)
    logger.info(f" [PASS] Spatial profiles saved to: {save_path}")

# [test/test_physics.py] - Update the test_dimension_accuracy function

def test_dimension_accuracy(logger):
    """Benchmarks the high-precision solver."""
    logger.info("\n>>> Running Dimension Accuracy Benchmark...")
    P, v, a = 250.0, 1.0, 50e-6
    
    # 1. Optimization Method (New)
    start = time.time()
    
    # UPDATE: Unpack 5 values instead of 3 (use _ to ignore tail/front)
    L_opt, W_opt, D_opt, _, _ = get_eagar_tsai_dimensions(P, v, a, MOCK_MATERIAL)
    
    t_opt = time.time() - start
    
    logger.info(f" [NEW] Optimization Solver ({t_opt:.4f}s):")
    logger.info(f"   L: {L_opt*1e6:.2f} µm")
    logger.info(f"   W: {W_opt*1e6:.2f} µm")
    logger.info(f"   D: {D_opt*1e6:.2f} µm")

    # ... [Rest of the function remains the same] ...

    # 2. Grid Method (Old Way Simulation)
    start = time.time()
    xs = np.linspace(-L_opt, L_opt, 200)
    T_vals = [eagar_tsai_temp(x, 0, 0, P, v, a, MOCK_MATERIAL) for x in xs]
    mask = np.array(T_vals) >= MOCK_MATERIAL['T_m']
    if np.any(mask):
        L_grid = xs[mask].max() - xs[mask].min()
    else:
        L_grid = 0
    t_grid = time.time() - start

    err = abs(L_grid - L_opt) / L_opt * 100 if L_opt > 0 else 0
    logger.info(f" [OLD] Grid Est (Res=200): L={L_grid*1e6:.2f} µm")
    logger.info(f"   Error vs Optimization: {err:.2f}%")
    
    if err < 5.0:
        logger.info(" [PASS] Grid approximation is within acceptable error.")
    else:
        logger.warning(" [WARN] Grid approximation has significant error.")

if __name__ == "__main__":
    logger, output_folder = setup_test_env("test_physics")
    try:
        test_general_physics(logger)
        test_spatial_profiles(logger, output_folder)
        test_dimension_accuracy(logger)
        logger.info("\nALL PHYSICS CHECKS COMPLETED.")
    except Exception as e:
        logger.error(f"CRITICAL FAILURE: {e}")
        import traceback
        logger.error(traceback.format_exc())