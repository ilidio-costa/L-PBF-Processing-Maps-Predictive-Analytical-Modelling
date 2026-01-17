import matplotlib.pyplot as plt
import os

# Import the setup function from your local utils
from utils import setup_test_env

# Import the functions to test
from src.plots import (
    top_view_eagar_tsai, 
    side_view_eagar_tsai,
    plot_process_grid_views,
    plot_defect_map,
    top_view_rubenchik,
    side_view_rubenchik
)

def test_plot_generation():
    # 1. Setup Environment (Logs & Output Folder)
    logger, output_folder = setup_test_env("test_plots")
    
    # 2. Define Mock Data (So we don't depend on loading external files)
    mock_material = {
        "name": "TestMaterial",
        "A": 0.4, "k": 20.0, "alpha": 1.0e-5,
        "rho": 4500.0, "C_p": 500.0, "T_m": 1900.0, "T_b": 3000.0
    }
    P, v, a = 200.0, 1.0, 50e-6

    logger.info(f"Testing with parameters: P={P}W, v={v}m/s, a={a*1e6}um")

    # --- Test 1: Top View ---
    logger.info("Generating Top View Plot...")
    try:
        # Calls the function with ALL required arguments
        fig1 = top_view_eagar_tsai(P, v, a, mock_material, resolution=50)
        
        save_path = os.path.join(output_folder, "top_view_eagar_tsai.png")
        fig1.savefig(save_path)
        plt.close(fig1) # Close memory
        logger.info(f" [PASS] Saved to {save_path}")
    except Exception as e:
        logger.error(f" [FAIL] Top view failed: {e}")

    # --- Test 2: Side View ---
    logger.info("Generating Side View Plot...")
    try:
        fig2 = side_view_eagar_tsai(P, v, a, mock_material, resolution=50)
        
        save_path = os.path.join(output_folder, "side_view_eagar_tsai.png")
        fig2.savefig(save_path)
        plt.close(fig2)
        logger.info(f" [PASS] Saved to {save_path}")
    except Exception as e:
        logger.error(f" [FAIL] Side view failed: {e}")

    

    # --- Test 3: Defect Map ---
    logger.info("Generating Defect Map...")
    try:
        x_range = (0.3, 3.0)
        y_range = (50, 500)
        fixed_params = {'a': a}
        
        # CAPTURE THE RETURNED FIGURE
        fig3 = plot_defect_map('v', 'P', x_range, y_range, fixed_params, mock_material, resolution=25)
        
        save_path = os.path.join(output_folder, "defect_map.png")
        fig3.savefig(save_path)
        plt.close(fig3)
        logger.info(f" [PASS] Saved to {save_path}")
    except Exception as e:
        logger.error(f" [FAIL] Defect map failed: {e}")


def test_rubenchik_plots():
    """Test for Rubenchik Top and Side View plots."""
    # 1. Setup Environment
    logger, output_folder = setup_test_env("test_plots_rubenchik")
    
    # 2. Define Mock Data
    mock_material = {
        "name": "TestMaterial",
        "A": 0.4, "k": 20.0, "alpha": 1.0e-5,
        "rho": 4500.0, "C_p": 500.0, "T_m": 1900.0, "T_b": 3000.0
    }
    P, v, a = 200.0, 1.0, 50e-6

    logger.info(f"Testing Rubenchik with parameters: P={P}W, v={v}m/s, a={a*1e6}um")

    # --- Test 1: Rubenchik Top View ---
    logger.info("Generating Rubenchik Top View...")
    try:
        # Calls the function from src/plots.py
        fig1 = top_view_rubenchik(P, v, a, mock_material, resolution=50)
        
        save_path = os.path.join(output_folder, "rubenchik_top_view.png")
        fig1.savefig(save_path)
        plt.close(fig1) # Close memory
        logger.info(f" [PASS] Saved to {save_path}")
    except Exception as e:
        logger.error(f" [FAIL] Rubenchik Top view failed: {e}")
        import traceback
        logger.error(traceback.format_exc())

    # --- Test 2: Rubenchik Side View ---
    logger.info("Generating Rubenchik Side View...")
    try:
        # Calls the function from src/plots.py
        fig2 = side_view_rubenchik(P, v, a, mock_material, resolution=50)
        
        save_path = os.path.join(output_folder, "rubenchik_side_view.png")
        fig2.savefig(save_path)
        plt.close(fig2)
        logger.info(f" [PASS] Saved to {save_path}")
    except Exception as e:
        logger.error(f" [FAIL] Rubenchik Side view failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
   

def test_grid_view():
    """Runs the NEW Grid Process Map test."""
    logger, output_folder = setup_test_env("test_plots_grid")
    
    mock_material = {
        "name": "TestMaterial",
        "A": 0.4, "k": 20.0, "alpha": 1.0e-5,
        "rho": 4500.0, "C_p": 500.0, "T_m": 1900.0, "T_b": 3000.0
    }
    a = 50e-6
    
    # Define Process Window
    P_range = [150.0, 250.0, 350.0]
    v_range = [1.5, 1.0, 0.5] # Descending velocity for rows
    
    logger.info(">>> Generating Grid Views (Top & Side)...")
    logger.info(f"Powers: {P_range}")
    logger.info(f"Velocities: {v_range}")

    try:
        # Lower resolution (60) for faster testing
        fig_top, fig_side = plot_process_grid_views(P_range, v_range, a, mock_material, resolution=60)
        
        top_path = os.path.join(output_folder, "grid_view_top.png")
        side_path = os.path.join(output_folder, "grid_view_side.png")
        
        fig_top.savefig(top_path, dpi=120)
        fig_side.savefig(side_path, dpi=120)
        
        plt.close(fig_top)
        plt.close(fig_side)
        
        logger.info(f" [PASS] Saved Grid Top View to {top_path}")
        logger.info(f" [PASS] Saved Grid Side View to {side_path}")
    except Exception as e:
        logger.error(f" [FAIL] Grid generation failed: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    test_plot_generation()
    #test_grid_view()
    test_rubenchik_plots()