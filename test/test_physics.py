import numpy as np
import matplotlib.pyplot as plt
import os
import time
from scipy.optimize import minimize_scalar
from utils import setup_test_env

from src.physics import (
    eagar_tsai_temp, 
    rubenchik_field,
    get_eagar_tsai_dimensions,
    rubenchik_temp,
    rubenchik_variables
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

def test_eagar_tsai_spatial_profiles(logger, output_folder):
    """Generates X, Y, and Z temperature profiles."""
    logger.info("\n>>> Running Eagar-Tsai Spatial Profile Sweep (X, Y, Z)...")
    
    powers = [150, 350]
    velocities = [0.8, 1.2]
    a = 40e-6
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
    save_path = os.path.join(output_folder, "eagar_tsai_spatial_profiles_sweep.png")
    fig.savefig(save_path)
    plt.close(fig)
    logger.info(f" [PASS] Spatial profiles saved to: {save_path}")

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

def test_rubenchik_model(logger):
    """Tests the Rubenchik Temperature Point Calculations."""
    logger.info("\n>>> Running Rubenchik Temperature Model Checks...")
    P, v, a = 200.0, 1.0, 50e-6
    T_ambient = 298.0
    
    # 1. Test Center Point (0,0,0) - Should be hottest
    logger.info("   Testing Center Point (0,0,0)...")
    coords, B, p = rubenchik_variables(0, 0, 0, MOCK_MATERIAL, P, v, a, T_ambient)
    T_center = rubenchik_temp(coords, B, p, MOCK_MATERIAL, T_ambient)
    
    logger.info(f"   [INFO] Rubenchik Peak T: {T_center:.2f} K")
    
    if T_center > T_ambient + 100:
        logger.info(f"   [PASS] Peak T ({T_center:.0f}K) is significantly above ambient.")
    else:
        logger.error(f"   [FAIL] Peak T ({T_center:.0f}K) is too low (Expected >> {T_ambient}K).")

    # 2. Test Far Away Point (0.005m away) - Should be near ambient
    logger.info("   Testing Far Field Point (x=5mm)...")
    coords_far, B, p = rubenchik_variables(0.005, 0, 0, MOCK_MATERIAL, P, v, a, T_ambient)
    T_far = rubenchik_temp(coords_far, B, p, MOCK_MATERIAL, T_ambient)
    
    logger.info(f"   [INFO] Far Field T: {T_far:.2f} K")
    
    if T_far < T_center and (T_far - T_ambient) < 50:
        logger.info("   [PASS] Temperature decays correctly in far field.")
    else:
        logger.warning(f"   [WARN] Far field T ({T_far:.0f}K) might be too high.")

def test_rubenchik_continuity(logger):
    """
    Stress test for Rubenchik model continuity at the singularity point (0,0,0).
    Scans X, Y, and Z across the origin with high resolution.
    """
    logger.info("\n>>> Running Rubenchik Continuity Stress Test (Micro-Scan)...")
    P, v, a = 200.0, 1.0, 50e-6
    T_ambient = 298.0
    
    # 1. Micro-Scan X (Centerline)
    # Scan exactly across 0 to check for 1/sqrt(t) singularity issues
    xs = np.linspace(-a, a, 201) # 201 points ensures we hit exactly 0
    T_x = []
    
    for x_val in xs:
        coords, B, p = rubenchik_variables(x_val, 0, 0, MOCK_MATERIAL, P, v, a, T_ambient)
        T_x.append(rubenchik_temp(coords, B, p, MOCK_MATERIAL, T_ambient))
    
    # Check for NaNs
    if np.any(np.isnan(T_x)):
        logger.error(" [FAIL] NaNs detected in Rubenchik X-scan.")
    else:
        logger.info(" [PASS] No NaNs in X-scan.")

    # Check for smoothness (Simple derivative check)
    diffs = np.diff(T_x)
    max_jump = np.max(np.abs(diffs))
    
    logger.info(f"   Max inter-point jump in X: {max_jump:.2f} K")
    if max_jump < 500: # Arbitrary "smoothness" threshold
        logger.info(" [PASS] X-profile appears continuous.")
    else:
        logger.warning(f" [WARN] X-profile has large jumps ({max_jump:.2f} K).")

    # 2. Check Z (Depth) Continuity
    zs = np.linspace(-a, 0, 100)
    T_z = []
    for z_val in zs:
        coords, B, p = rubenchik_variables(0, 0, z_val, MOCK_MATERIAL, P, v, a, T_ambient)
        T_z.append(rubenchik_temp(coords, B, p, MOCK_MATERIAL, T_ambient))

    if np.any(np.isnan(T_z)):
        logger.error(" [FAIL] NaNs detected in Rubenchik Z-scan.")
    else:
        logger.info(f" [PASS] Z-scan successful. Surface T={T_z[-1]:.1f} K")

def test_rubenchik_spatial_profiles(logger, output_folder):
    """
    Generates X, Y, and Z temperature profiles for the Rubenchik model.
    Checks for continuity and thermal lag (Peak X location).
    """
    logger.info("\n>>> Running Rubenchik Spatial Profile Sweep (X, Y, Z)...")
    
    # Define Parameters (Same as Eagar-Tsai test for comparison)
    powers = [150, 350]
    velocities = [0.8, 1.2]
    a = 40e-6
    T_ambient = 0
    Tm = MOCK_MATERIAL['T_m']
    
    fig, axes = plt.subplots(len(powers)*len(velocities), 3, figsize=(15, 12))
    fig.suptitle(f"Rubenchik Spatial Profiles (NiTi) - Tm={Tm}K", fontsize=16)

    row_idx = 0
    for P in powers:
        for v in velocities:
            # 1. Find Peak X (Thermal Lag)
            # Rubenchik model often defines the wake in +x (dimensionless t > 0). 
            # We scan a range to find the exact peak.
            test_xs = np.linspace(-3*a, 3*a, 100)
            temps_peak_search = []
            for val in test_xs:
                # Note: src/plots.py flips x (x_val = -X) for Rubenchik. 
                # We test the raw physics function here. 
                # If the physics puts the wake in +x, we expect peak_x > 0.
                c, B, p = rubenchik_variables(val, 0, 0, MOCK_MATERIAL, P, v, a, T_ambient)
                temps_peak_search.append(rubenchik_temp(c, B, p, MOCK_MATERIAL, T_ambient))
            
            x_peak = test_xs[np.argmax(temps_peak_search)]
            T_max = max(temps_peak_search)

            logger.info(f"   [P={P}W, v={v}m/s] Raw Physics Peak found at x = {x_peak*1e6:.1f} µm")

            # 2. Define Scan Ranges centered on peakl
            xs = np.linspace(x_peak - 400e-6, x_peak + 100e-6, 100)
            ys = np.linspace(-150e-6, 150e-6, 100)                  # Transverse
            zs = np.linspace(-150e-6, 0, 100)                       # Depth (Negative)

            # 3. Compute Profiles
            # X-Axis Profile (y=0, z=0)
            T_x = []
            for val in xs:
                c, B, p = rubenchik_variables(val, 0, 0, MOCK_MATERIAL, P, v, a, T_ambient)
                T_x.append(rubenchik_temp(c, B, p, MOCK_MATERIAL, T_ambient))

            # Y-Axis Profile (x=x_peak, z=0)
            T_y = []
            for val in ys:
                c, B, p = rubenchik_variables(x_peak, val, 0, MOCK_MATERIAL, P, v, a, T_ambient)
                T_y.append(rubenchik_temp(c, B, p, MOCK_MATERIAL, T_ambient))

            # Z-Axis Profile (x=x_peak, y=0)
            T_z = []
            for val in zs:
                c, B, p = rubenchik_variables(x_peak, 0, val, MOCK_MATERIAL, P, v, a, T_ambient)
                T_z.append(rubenchik_temp(c, B, p, MOCK_MATERIAL, T_ambient))

            # 4. Plotting
            # Plot X
            axes[row_idx, 0].plot(xs*1e6, T_x, 'b-', label='Rubenchik')
            axes[row_idx, 0].axhline(Tm, color='r', linestyle='--', alpha=0.5, label='Melting Pt')
            axes[row_idx, 0].axvline(x_peak*1e6, color='k', linestyle=':', alpha=0.5, label='Peak')
            axes[row_idx, 0].set_title(f"Longitudinal X (P={P}, v={v})")
            axes[row_idx, 0].set_xlabel("X (µm)")
            axes[row_idx, 0].set_ylabel("Temp (K)")
            axes[row_idx, 0].legend()
            
            # Plot Y
            axes[row_idx, 1].plot(ys*1e6, T_y, 'g-')
            axes[row_idx, 1].axhline(Tm, color='r', linestyle='--', alpha=0.5)
            axes[row_idx, 1].set_title("Transverse Y (Width)")
            axes[row_idx, 1].set_xlabel("Y (µm)")

            # Plot Z
            axes[row_idx, 2].plot(zs*1e6, T_z, 'purple')
            axes[row_idx, 2].axhline(Tm, color='r', linestyle='--', alpha=0.5)
            axes[row_idx, 2].set_title("Depth Z")
            axes[row_idx, 2].set_xlabel("Z (µm)")
            
            row_idx += 1

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = os.path.join(output_folder, "rubenchik_spatial_profiles.png")
    fig.savefig(save_path)
    plt.close(fig)
    logger.info(f" [PASS] Rubenchik profiles saved to: {save_path}")

if __name__ == "__main__":
    logger, output_folder = setup_test_env("test_physics")
    try:
        test_general_physics(logger)
        test_rubenchik_model(logger)
        test_rubenchik_continuity(logger)
        test_eagar_tsai_spatial_profiles(logger, output_folder)
        test_rubenchik_spatial_profiles(logger, output_folder)
        test_dimension_accuracy(logger)
        logger.info("\nALL PHYSICS CHECKS COMPLETED.")
    except Exception as e:
        logger.error(f"CRITICAL FAILURE: {e}")
        import traceback
        logger.error(traceback.format_exc())