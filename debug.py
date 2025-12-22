from librosa import resample
from semver import process
from src.plots import length_depth_eagar_tsai, length_width_eagar_tsai,gaussian_laser,length_depth_rubenchik,length_width_rubenchik, plot_melt_pool_dimensions, plot_defect_map
from src.data_loader import load_material

# Define standard Steel properties
mat = load_material("NiTi.json")
process_P = 250   
process_v = 1.25  
process_a = 40e-6   


#x_range=(-1e-3, 0.07e-3)
#y_range=(-0.15e-3, 0.15e-3)
x_range=(0.3,2.8)
y_range=(50,400)
z_depth=-0.175e-3
T_ambient=0
resolution=200
fixed_params={'a': 40e-6}
# Run the plot
#length_width_eagar_tsai(process_P, process_v, process_a, mat, x_range, y_range, resolution)
#length_depth_eagar_tsai(process_P, process_v, process_a, mat, x_range, z_depth, resolution)

#length_width_rubenchik(process_P, process_v, process_a, mat, T_ambient, x_range, y_range, resolution)
#length_depth_rubenchik(process_P, process_v, process_a, mat, T_ambient, x_range, z_depth, resolution)

#plot_melt_pool_dimensions('v', 'P', x_range, y_range, fixed_params, mat, T_ambient=0, use_gladush=False, resolution=100)

plot_defect_map('v', 'P', x_range, y_range, fixed_params, mat, layer_t=30e-6, hatch_s=80e-6, resolution=300)

#gaussian_laser()