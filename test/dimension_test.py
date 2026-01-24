import time
import numpy as np
from utils import setup_test_env  # Importing this first sets up sys.path automatically

# Now we can import from src without errors
from src.physics import get_eagar_tsai_dimensions, get_melt_pool_dimensions

# --- MOCK MATERIAL (NiTi) ---
MOCK_MATERIAL ={
            "name": "NiTi",
            "rho": 6100,
            "C_p": 510,
            "k": 4.4,
            "T_b": 3033,
            "T_m": 1583,
            "A": 0.32,
            "alpha": 8e-6
            }

def compare_melt_pool_algorithms(logger):
    logger.info("="*60)
    logger.info("      COMPARISON: GLOBAL SEARCH (Old) vs 1D ROOT (New)")
    logger.info("="*60)

    # Test Parameters
    P = 200.0   # Power [W]
    v = 1.0     # Velocity [m/s]
    a = 50e-6   # Beam Radius [m]
    
    logger.info(f"Parameters: P={P}W, v={v}m/s, a={a*1e6}µm\n")

    # --- 1. Old Method (Global Search) ---
    start_time = time.time()
    res_old = get_eagar_tsai_dimensions(P, v, a, MOCK_MATERIAL, resolution=200)
    end_time = time.time()
    
    L_old, W_old, D_old = res_old[0], res_old[1], res_old[2]
    time_old = end_time - start_time

    # --- 2. New Method (1D Root Finding) ---
    start_time = time.time()
    res_new = get_melt_pool_dimensions(P, v, a, MOCK_MATERIAL)
    end_time = time.time()
    
    L_new, W_new, D_new = res_new[0], res_new[1], res_new[2]
    time_new = end_time - start_time

    # --- 3. Comparison Output ---
    # Formatting helper for cleaner logs
    header = f"{'METRIC':<10} | {'OLD (Global)':<15} | {'NEW (1D Root)':<15} | {'DIFF (%)':<10}"
    logger.info(header)
    logger.info("-" * 60)
    
    def log_row(label, val_old, val_new):
        if val_old == 0:
            diff = 0.0 if val_new == 0 else 100.0
        else:
            diff = abs(val_new - val_old) / val_old * 100
        logger.info(f"{label:<10} | {val_old*1e6:>12.2f} µm | {val_new*1e6:>12.2f} µm | {diff:>9.2f}%")

    log_row("Length", L_old, L_new)
    log_row("Width", W_old, W_new)
    log_row("Depth", D_old, D_new)
    
    logger.info("-" * 60)
    
    speedup = time_old / time_new if time_new > 0 else 0
    logger.info(f"{'Time':<10} | {time_old*1000:>12.2f} ms | {time_new*1000:>12.2f} ms | {speedup:>9.1f}x Faster")
    logger.info("=" * 60)

    # Verification Logic
    if abs(L_new - L_old) / (L_old if L_old > 0 else 1) < 0.05:
        logger.info("\n[PASS] New method is within 5% accuracy of the old method.")
    else:
        logger.warning("\n[WARN] Significant discrepancy detected (>5%). Check convergence.")

if __name__ == "__main__":
    # Initialize Logger and Environment
    # This creates 'test/logs/dimension_test.log'
    logger, _ = setup_test_env("dimension_test")
    
    try:
        compare_melt_pool_algorithms(logger)
        logger.info("\nTEST COMPLETED SUCCESSFULLY.")
    except Exception as e:
        logger.error(f"CRITICAL FAILURE: {e}")
        import traceback
        logger.error(traceback.format_exc())