import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt

# --- Setup Paths ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(BASE_DIR)

# IMPORT THE NEW FUNCTION HERE
from src.plots import (
    top_view_eagar_tsai, side_view_eagar_tsai,
    top_view_rubenchik, side_view_rubenchik,
    plot_process_et_grid_views, plot_process_r_grid_views,
    plot_melt_pool_dimensions
)

# --- Setup Output Directory ---
out_dir = os.path.join(BASE_DIR, 'test', 'output')
os.makedirs(out_dir, exist_ok=True)

def load_test_material():
    """Loads NiTi from the materials folder for testing."""
    mat_path = os.path.join(BASE_DIR, 'materials', 'NiTi.json')
    if not os.path.exists(mat_path):
        print(f"Error: Could not find material file at {mat_path}")
        sys.exit(1)
        
    with open(mat_path, 'r') as f:
        mat = json.load(f)
        if 'alpha' not in mat:
            mat['alpha'] = mat['k'] / (mat['rho'] * mat['C_p'])
        return mat

def run_all_plot_tests():
    mat = load_test_material()
    a = 40e-6
    P_nominal, v_nominal = 250.0, 1.25
    
    print("===================================================")
    print("          TESTING PLOTTING CAPABILITIES")
    print("===================================================\n")

    # ---------------------------------------------------------
    # 1. Single Melt Pool Views
    # ---------------------------------------------------------
    print("1/3 Generating Single Melt Pool Views...")
    
    # Eagar-Tsai
    fig_et_top = top_view_eagar_tsai(P_nominal, v_nominal, a, mat)
    fig_et_top.savefig(os.path.join(out_dir, 'test_et_top_view.png'), dpi=300)
    plt.close(fig_et_top)
    
    fig_et_side = side_view_eagar_tsai(P_nominal, v_nominal, a, mat)
    fig_et_side.savefig(os.path.join(out_dir, 'test_et_side_view.png'), dpi=300)
    plt.close(fig_et_side)  

    # Rubenchik
    fig_rub_top = top_view_rubenchik(P_nominal, v_nominal, a, mat)
    fig_rub_top.savefig(os.path.join(out_dir, 'test_rub_top_view.png'), dpi=300)
    plt.close(fig_rub_top)
    
    fig_rub_side = side_view_rubenchik(P_nominal, v_nominal, a, mat)
    fig_rub_side.savefig(os.path.join(out_dir, 'test_rub_side_view.png'), dpi=300)
    plt.close(fig_rub_side)
    
    # ---------------------------------------------------------
    # 2. Master Grid Views
    # ---------------------------------------------------------
    print("2/3 Generating Master Grid Views...")
    P_list = [50, 89, 158, 281, 500]
    v_list = [0.35, 0.62, 1.11, 1.97, 3.5]

    # Eagar-Tsai Grids
    fig_old_et_grid_top, fig_old_et_grid_side = plot_process_et_grid_views(P_list, v_list, a, mat, 0, 100, True )
    fig_old_et_grid_top.savefig(os.path.join(out_dir, 'test_old_et_MASTER_grid_top.png'), dpi=300)
    fig_old_et_grid_side.savefig(os.path.join(out_dir, 'test_old_et_MASTER_grid_side.png'), dpi=300)
    plt.close(fig_old_et_grid_top)
    plt.close(fig_old_et_grid_side)
    
    # Rubenchik Grids
    fig_old_rub_grid_top, fig_old_rub_grid_side = plot_process_r_grid_views(P_list, v_list, a, mat, 0, 100, True)
    fig_old_rub_grid_top.savefig(os.path.join(out_dir, 'test_old_rub_MASTER_grid_top.png'), dpi=300)
    fig_old_rub_grid_side.savefig(os.path.join(out_dir, 'test_old_rub_MASTER_grid_side.png'), dpi=300)
    plt.close(fig_old_rub_grid_top)
    plt.close(fig_old_rub_grid_side)

    # ---------------------------------------------------------
    # 3. Melt Pool Dimension Contour Maps (Printability Maps)
    # ---------------------------------------------------------
    print("3/3 Generating Melt Pool Dimension Contour Maps...")
    
    x_var = 'v'
    y_var = 'P'
    x_range = (0.1, 3.5) # Scanning Velocity Range: 0.1 to 3.5 m/s
    y_range = (50, 500)  # Laser Power Range: 50 to 500 W
    fixed_params = {'a': a} # Fix the spot size radius
    test_resolution = 20 # Keep resolution low to ensure the test runs quickly

    # Test Eagar-Tsai + Gladush-Smurov
    print("    -> Plotting Eagar-Tsai Dimensions...")
    fig_dims_et = plot_melt_pool_dimensions(
        x_var, y_var, x_range, y_range, fixed_params, mat, 
        use_rubenchik=False, use_gladush=False, resolution=test_resolution
    )
    fig_dims_et.savefig(os.path.join(out_dir, 'test_printability_map_ET.png'), dpi=300, bbox_inches='tight')
    plt.close(fig_dims_et)

    # Test Rubenchik + Gladush-Smurov
    print("    -> Plotting Rubenchik Dimensions...")
    fig_dims_rub = plot_melt_pool_dimensions(
        x_var, y_var, x_range, y_range, fixed_params, mat, 
        use_rubenchik=True, use_gladush=True, resolution=test_resolution
    )
    fig_dims_rub.savefig(os.path.join(out_dir, 'test_printability_map_RUB.png'), dpi=300, bbox_inches='tight')
    plt.close(fig_dims_rub)


    print("\n>>> ALL PLOTS GENERATED SUCCESSFULLY!")
    print(f">>> Check the '{out_dir}' folder to view the output images.")

if __name__ == '__main__':
    run_all_plot_tests()