import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt

from src.physics import compute_printability_map
from src.plots import plot_deterministic_map
from src.data_loader import load_material

def test_compute_printability_map():
    print("Testing compute_printability_map...")
    
    Power_range = (50, 350)      
    Scan_Speed_range = (0.1, 2.5) 
    resolution = 35              

    material = load_material("Al6061.json")
    
    # Updated to include T_ambient
    process_parameters = {'t': 30e-6, 'h': 100e-6, 'a': 25e-6, 'T_ambient': 443}
    
    active_defects = {
        'balling': 'ball01',
        'lof': 'lof01',
        'keyhole': 'key01'
    }

    P_grid, v_grid, defect_map = compute_printability_map(
        Power_range, 
        Scan_Speed_range, 
        material, 
        process_parameters, 
        resolution=resolution, 
        active_defects=active_defects
    )

    assert P_grid.shape == (resolution, resolution), f"Expected P_grid shape ({resolution}, {resolution}), got {P_grid.shape}"
    assert v_grid.shape == (resolution, resolution), f"Expected v_grid shape ({resolution}, {resolution}), got {v_grid.shape}"
    assert defect_map.shape == (resolution, resolution), f"Expected defect_map shape ({resolution}, {resolution}), got {defect_map.shape}"
    
    unique_values = np.unique(defect_map)
    for val in unique_values:
        assert val in [0, 1, 2, 3], f"Unexpected value {val} found in defect_map! Should only be 0, 1, 2, or 3."

    print("SUCCESS: compute_printability_map passed all checks!")
    return P_grid, v_grid, defect_map, material

def test_plot_deterministic_map(P_grid, v_grid, defect_map, material):
    print("Testing plot_deterministic_map...")
    try:
        fig = plot_deterministic_map(P_grid, v_grid, defect_map, material_name=material.get('name', 'NiTi'))
        assert isinstance(fig, plt.Figure), "The plotting function did not return a valid matplotlib Figure object."
        
        output_dir = os.path.join(os.path.dirname(__file__), 'output')
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'printability_map_Al6061.png'))
        plt.close(fig)
        
        print("SUCCESS")
    except Exception as e:
        print(f"FAILED: Plotting threw an exception: {e}")
        raise e

if __name__ == "__main__":
    P, v, d_map, mat = test_compute_printability_map()
    test_plot_deterministic_map(P, v, d_map, mat)