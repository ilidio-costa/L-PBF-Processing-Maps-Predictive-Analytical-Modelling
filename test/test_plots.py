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

mock_material = {
            "name": "NiTi",
            "rho": 6100,
            "C_p": 510,
            "k": 4.4,
            "T_b": 3033,
            "T_m": 1583,
            "A": 0.32,
            "alpha": 8e-6
            }

def test_plot_generation():
    # 1. Setup Environment (Logs & Output Folder)
    logger, output_folder = setup_test_env("test_plots")
    
    
    P, v, a = 200.0, 1.0, 40e-6

    logger.info(f"Testing with parameters: P={P}W, v={v}m/s, a={a*1e6}um")

    # --- Test 1: Top View ---
    logger.info("Generating Top View Plot...")
    try:
        # Calls the function with ALL required arguments
        fig1 = top_view_eagar_tsai(P, v, a, mock_material, resolution=200,remove_background=True)
        
        save_path = os.path.join(output_folder, "top_view_eagar_tsai.png")
        fig1.savefig(save_path)
        plt.close(fig1) # Close memory
        logger.info(f" [PASS] Saved to {save_path}")
    except Exception as e:
        logger.error(f" [FAIL] Top view failed: {e}")

    # --- Test 2: Side View ---
    logger.info("Generating Side View Plot...")
    try:
        fig2 = side_view_eagar_tsai(P, v, a, mock_material, resolution=200,remove_background=True)
        
        save_path = os.path.join(output_folder, "side_view_eagar_tsai.png")
        fig2.savefig(save_path)
        plt.close(fig2)
        logger.info(f" [PASS] Saved to {save_path}")
    except Exception as e:
        logger.error(f" [FAIL] Side view failed: {e}")

    

    # --- Test 3: Defect Map ---
    logger.info("Generating Defect Map...")
    try:
        x_range = (0.35, 3.5)
        y_range = (50, 500)
        fixed_params = {'a': a}
        
        # CAPTURE THE RETURNED FIGURE
        fig3 = plot_defect_map('v', 'P', x_range, y_range, fixed_params, mock_material, resolution=40)
        
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
    

    P, v, a = 200.0, 1.0, 40e-6

    logger.info(f"Testing Rubenchik with parameters: P={P}W, v={v}m/s, a={a*1e6}um")

    # --- Test 1: Rubenchik Top View ---
    logger.info("Generating Rubenchik Top View...")
    try:
        # Calls the function from src/plots.py
        fig1 = top_view_rubenchik(P, v, a, mock_material, resolution=50)
        
        save_path = os.path.join(output_folder, "top_view_rubenchik.png")
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
        
        save_path = os.path.join(output_folder, "side_view_rubenchik.png")
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
    

    a = 40e-6
    
    # Define Process Window
    P_range = [50, 88.9, 158.1, 281.2, 500]
    v_range = [0.35, 0.62, 1.11, 1.97, 3.5] # Descending velocity for rows
    
    logger.info(">>> Generating Grid Views (Top & Side)...")
    logger.info(f"Powers: {P_range}")
    logger.info(f"Velocities: {v_range}")

    try:
        # Lower resolution (60) for faster testing
        fig_top, fig_side = plot_process_grid_views(P_range, v_range, a, mock_material, resolution=80, remove_background=True)
        
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
    #test_rubenchik_plots()