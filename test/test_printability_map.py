import sys
import os
# This adds the parent directory (project root) to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt

# Import the functions we built
from src.physics import compute_printability_map
from src.plots import plot_deterministic_map
from src.data_loader import load_material

def test_compute_printability_map():
    """
    Tests the deterministic printability map generator to ensure it correctly
    creates the grids, loads the defect modules, and returns the expected matrix shape.
    """
    print("Testing compute_printability_map...")
    
    # 1. Define dummy test inputs (small resolution for fast testing)
    Power_range = (50, 350)      
    Scan_Speed_range = (0.1, 2.5) 
    resolution = 60              

    # Load material properties directly from the materials directory
    material = load_material("NiTi_Sheikh.json")
    
    # Standard L-PBF process parameters including laser spot size 'a'
    process_parameters = {'t': 30e-6, 'h': 80e-6, 'a': 40e-6}
    
    # Map to the exact files in your src/defects/ folder
    active_defects = {
        'balling': 'ball01',
        'lof': 'lof02',
        'keyhole': 'key01'
    }

    # 2. Run the function
    P_grid, v_grid, defect_map = compute_printability_map(
        Power_range, 
        Scan_Speed_range, 
        material, 
        process_parameters, 
        resolution=resolution, 
        active_defects=active_defects
    )

    # 3. Assertions to verify correct behavior
    assert P_grid.shape == (resolution, resolution), f"Expected P_grid shape ({resolution}, {resolution}), got {P_grid.shape}"
    assert v_grid.shape == (resolution, resolution), f"Expected v_grid shape ({resolution}, {resolution}), got {v_grid.shape}"
    assert defect_map.shape == (resolution, resolution), f"Expected defect_map shape ({resolution}, {resolution}), got {defect_map.shape}"
    
    # Check that the defect map only contains integers between 0 and 3
    unique_values = np.unique(defect_map)
    for val in unique_values:
        assert val in [0, 1, 2, 3], f"Unexpected value {val} found in defect_map! Should only be 0, 1, 2, or 3."

    print("SUCCESS: compute_printability_map passed all checks!")
    
    return P_grid, v_grid, defect_map, material

def test_plot_deterministic_map(P_grid, v_grid, defect_map, material):
    """
    Tests the plotting function to ensure it doesn't throw errors when 
    generating the matplotlib figure.
    """
    print("Testing plot_deterministic_map...")
    
    try:
        # Generate the figure (falling back to "NiTi" if the name key isn't in the JSON)
        fig = plot_deterministic_map(P_grid, v_grid, defect_map, material_name=material.get('name', 'NiTi'))
        
        # Ensure a figure object was returned
        assert isinstance(fig, plt.Figure), "The plotting function did not return a valid matplotlib Figure object."
        
        # Close the figure so it doesn't hang the test suite memory
        plt.savefig(os.path.join(os.path.dirname(__file__), 'output', 'printability_map_test.png'))
        plt.show()
        plt.close(fig)
        
        print("SUCCESS")
        
    except Exception as e:
        print(f"FAILED: Plotting threw an exception: {e}")
        raise e

if __name__ == "__main__":
    # Run the tests sequentially when executing this file directly
    P, v, d_map, mat = test_compute_printability_map()
    test_plot_deterministic_map(P, v, d_map, mat)