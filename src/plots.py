import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from .physics import get_temp_at_point, get_temp_at_point_substituted, rubenchik, get_temp_rubenchik, rubenchik_field, get_melt_depth_gladush, get_defect_masks

# Eagar-Tsai Melt Pool Plots
def length_depth_eagar_tsai(P, v, a, material, x_range=(-2e-3, 1e-3), z_depth=-1e-3, resolution=40):
    
    print(f"Calculating {resolution}x{resolution} grid...")

    xs = np.linspace(x_range[0], x_range[1], resolution)
    zs = np.linspace(z_depth, 0, resolution)
    X, Z = np.meshgrid(xs, zs)
    
    T_field = np.zeros_like(X)
    min_target_temp = material['T_m']  # Default to melting point if not specified

    for i in range(resolution):
        for j in range(resolution):
            x_val = X[i, j]
            z_val = Z[i, j] 
            
            # Calling the function imported from physics.py
            # Ensure get_temp_at_point handles base units (meters) correctly
            T_field[i, j] = get_temp_at_point_substituted(x_val, 0, z_val, P, v, a, material)

    # Plotting Logic
    plt.figure(figsize=(10, 6))
    max_temp = np.max(T_field)
    
    melt_mask = T_field >= min_target_temp
    if np.any(melt_mask):
        x_melt = X[melt_mask]
        z_melt = Z[melt_mask]
        pool_length = (np.max(x_melt) - np.min(x_melt)) * 1e6
        pool_depth = (np.max(z_melt) - np.min(z_melt)) * 1e6
        dim_label = f"L: {pool_length:.1f}µm | D: {pool_depth:.1f}µm"
    else:
        dim_label = "No Melt Pool"

    if max_temp < min_target_temp:
        levels = np.linspace(293, max_temp, 20)
    else:
        levels = np.linspace(min_target_temp, max_temp, 50)

    # CHANGE 1: Multiply X and Z by 1e6 instead of 1e3 to convert meters to microns
    contour = plt.contourf(X * 1e6, Z * 1e6, T_field, levels=levels, cmap='inferno', extend='max')
    plt.contour(X * 1e6, Z * 1e6, T_field, levels=[min_target_temp], colors='cyan', linewidths=2, linestyles='dashed')

    plt.colorbar(contour, label='Temperature (K)')
    
    # CHANGE 2: Update labels to use microns (mu m)
    # Using raw string (r'...') allows latex formatting for the greek letter mu
    plt.xlabel(r'Distance ($\mu m$)')
    plt.ylabel(r'Depth Z ($\mu m$)')
    
    plt.title(f'Melt Pool XZ View (y=0)\n{dim_label}\nP={P}W, v={v*1000}mm/s')
    
    # CHANGE 3: Update limit scaling to 1e6
    plt.ylim(z_depth * 1e6, 0)
    
    plt.grid(True, alpha=0.2)
    plt.show()

def length_width_eagar_tsai(P, v, a, material, x_range=(-0.5e-3, 0.5e-3), y_range=(-0.2e-3, 0.2e-3), resolution=40):
    
    print(f"Calculating {resolution}x{resolution} grid (Top View)...")

    # 1. Setup Grid
    xs = np.linspace(x_range[0], x_range[1], resolution)
    ys = np.linspace(y_range[0], y_range[1], resolution)
    X, Y = np.meshgrid(xs, ys)
    
    T_field = np.zeros_like(X)
    min_target_temp = material['T_m'] 

    # 2. Calculate Temperature Field
    for i in range(resolution):
        for j in range(resolution):
            x_val = X[i, j]
            y_val = Y[i, j] 
            # Z is fixed at 0 for top-down view
            T_field[i, j] = get_temp_at_point_substituted(x_val, y_val, 0, P, v, a, material)

    # 3. Calculate Dimensions for the Title
    melt_mask = T_field >= min_target_temp
    if np.any(melt_mask):
        x_melt = X[melt_mask]
        y_melt = Y[melt_mask]
        pool_length = (np.max(x_melt) - np.min(x_melt)) * 1e6
        pool_width = (np.max(y_melt) - np.min(y_melt)) * 1e6
        dim_label = f"L: {pool_length:.1f}µm | W: {pool_width:.1f}µm"
    else:
        dim_label = "No Melt Pool"

    # 4. Plotting Logic (Using fig, ax)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    max_temp = np.max(T_field)
    ambient = 293.0
    
    # Robust levels logic
    if max_temp <= ambient:
        levels = np.linspace(ambient, ambient+10, 20)
    elif max_temp < min_target_temp:
        levels = np.linspace(ambient, max_temp, 20)
    else:
        levels = np.linspace(min_target_temp, max_temp, 50)

    # Plot Contours (Scale to microns)
    contour = ax.contourf(X * 1e6, Y * 1e6, T_field, levels=levels, cmap='inferno', extend='max')
    
    # Plot Melt Boundary
    if max_temp >= min_target_temp:
        ax.contour(X * 1e6, Y * 1e6, T_field, levels=[min_target_temp], colors='cyan', linewidths=2, linestyles='dashed')

    # Add Colorbar
    cbar = fig.colorbar(contour, ax=ax)
    cbar.set_label('Temperature (K)')

    # Labels and Title
    ax.set_xlabel(r'Length X ($\mu m$)')
    ax.set_ylabel(r'Width Y ($\mu m$)')
    ax.set_title(f'Melt Pool XY View (z=0)\n{dim_label}\nP={P}W, v={v*1000}mm/s')
    
    ax.grid(True, alpha=0.2)
    ax.set_aspect('equal') # Important for top-down view
    
    # Return the figure object
    return fig

# Rubenchik Melt Pool Plots
def length_depth_rubenchik(P, v, a, material, T_ambient=0, x_range=(-2e-3, 1e-3), z_depth=-1e-3, resolution=40):
    
    print(f"Calculating {resolution}x{resolution} grid...")

    xs = np.linspace(x_range[0], x_range[1], resolution)
    zs = np.linspace(z_depth, 0, resolution)
    X, Z = np.meshgrid(xs, zs)
    
    T_field = np.zeros_like(X)
    min_target_temp = material['T_m']  # Default to melting point if not specified

    for i in range(resolution):
        for j in range(resolution):
            x_val = -X[i, j]
            z_val = Z[i, j] 
            
            # In src/plots.py, inside the nested loops for Rubenchik plots:
            coords, B, p = rubenchik(x_val, 0, z_val, material, P, v, a, T_ambient)
            # Ensure this now receives the Kelvin value from the updated physics function
            T_field[i, j] = get_temp_rubenchik(coords, B, p, material, T_ambient)

    # Plotting Logic
    plt.figure(figsize=(10, 6))
    max_temp = np.max(T_field)
    
    melt_mask = T_field >= min_target_temp
    if np.any(melt_mask):
        x_melt = X[melt_mask]
        z_melt = Z[melt_mask]
        pool_length = (np.max(x_melt) - np.min(x_melt)) * 1e6
        pool_depth = (np.max(z_melt) - np.min(z_melt)) * 1e6
        dim_label = f"L: {pool_length:.1f}µm | D: {pool_depth:.1f}µm"
    else:
        dim_label = "No Melt Pool"

    if max_temp < min_target_temp:
        levels = np.linspace(293, max_temp, 20)
    else:
        levels = np.linspace(min_target_temp, max_temp, 50)

    # CHANGE 1: Multiply X and Z by 1e6 instead of 1e3 to convert meters to microns
    contour = plt.contourf(X * 1e6, Z * 1e6, T_field, levels=levels, cmap='inferno', extend='max')
    plt.contour(X * 1e6, Z * 1e6, T_field, levels=[min_target_temp], colors='cyan', linewidths=2, linestyles='dashed')

    plt.colorbar(contour, label='Temperature (K)')
    
    # CHANGE 2: Update labels to use microns (mu m)
    # Using raw string (r'...') allows latex formatting for the greek letter mu
    plt.xlabel(r'Distance ($\mu m$)')
    plt.ylabel(r'Depth Z ($\mu m$)')
    
    plt.title(f'Melt Pool XZ View (y=0)\n{dim_label}\nP={P}W, v={v*1000}mm/s')
    
    # CHANGE 3: Update limit scaling to 1e6
    plt.ylim(z_depth * 1e6, 0)
    
    plt.grid(True, alpha=0.2)
    plt.show()

def length_width_rubenchik(P, v, a, material, T_ambient=0, x_range=(-0.5e-3, 0.5e-3), y_range=(-0.2e-3, 0.2e-3), resolution=40):
    
    print(f"Calculating {resolution}x{resolution} grid (Top View)...")

    # 1. Setup Grid
    xs = np.linspace(x_range[0], x_range[1], resolution)
    ys = np.linspace(y_range[0], y_range[1], resolution)
    X, Y = np.meshgrid(xs, ys)
    
    T_field = np.zeros_like(X)
    min_target_temp = material['T_m'] 

    # 2. Calculate Temperature Field
    for i in range(resolution):
        for j in range(resolution):
            x_val = -X[i, j]
            y_val = Y[i, j] 

            # In src/plots.py, inside the nested loops for Rubenchik plots:
            coords, B, p = rubenchik(x_val, y_val, 0, material, P, v, a, T_ambient)
            # Ensure this now receives the Kelvin value from the updated physics function
            T_field[i, j] = get_temp_rubenchik(coords, B, p, material, T_ambient)


    # 3. Calculate Dimensions for the Title
    melt_mask = T_field >= min_target_temp
    if np.any(melt_mask):
        x_melt = X[melt_mask]
        y_melt = Y[melt_mask]
        pool_length = (np.max(x_melt) - np.min(x_melt)) * 1e6
        pool_width = (np.max(y_melt) - np.min(y_melt)) * 1e6
        dim_label = f"L: {pool_length:.1f}µm | W: {pool_width:.1f}µm"
    else:
        dim_label = "No Melt Pool"

    # 4. Plotting Logic (Using fig, ax)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    max_temp = np.max(T_field)
    ambient = 293.0
    
    # Robust levels logic
    if max_temp <= ambient:
        levels = np.linspace(ambient, ambient+10, 20)
    elif max_temp < min_target_temp:
        levels = np.linspace(ambient, max_temp, 20)
    else:
        levels = np.linspace(min_target_temp, max_temp, 50)

    # Plot Contours (Scale to microns)
    contour = ax.contourf(X * 1e6, Y * 1e6, T_field, levels=levels, cmap='inferno', extend='max')
    
    # Plot Melt Boundary
    if max_temp >= min_target_temp:
        ax.contour(X * 1e6, Y * 1e6, T_field, levels=[min_target_temp], colors='cyan', linewidths=2, linestyles='dashed')

    # Add Colorbar
    cbar = fig.colorbar(contour, ax=ax)
    cbar.set_label('Temperature (K)')

    # Labels and Title
    ax.set_xlabel(r'Length X ($\mu m$)')
    ax.set_ylabel(r'Width Y ($\mu m$)')
    ax.set_title(f'Melt Pool XY View (z=0)\n{dim_label}\nP={P}W, v={v*1000}mm/s')
    
    ax.grid(True, alpha=0.2)
    ax.set_aspect('equal') # Important for top-down view
    
    # Return the figure object
    return fig


# Processing Map Melt Pool Dimension Plotting
def plot_melt_pool_dimensions(x_var, y_var, x_range, y_range, fixed_params, material, T_ambient=298, use_gladush=True, resolution=100):
    """
    Plots formatted contour maps for Melt Pool Length, Width, and Depth.
    Output units are converted to microns for visualization.
    """
    
    # 1. Setup Grid
    x = np.linspace(x_range[0], x_range[1], resolution)
    y = np.linspace(y_range[0], y_range[1], resolution)
    X, Y = np.meshgrid(x, y)
    
    # 2. Map X, Y, and Fixed values to P, v, a
    mapping = {x_var: X, y_var: Y}
    for key, val in fixed_params.items():
        mapping[key] = val
        
    P_grid = mapping['P']
    v_grid = mapping['v']
    a_grid = mapping['a']

    # 3. Calculate Dimensions (in meters)
    L_m, W_m, D_rubenchik_m = rubenchik_field(P_grid, v_grid, a_grid, material, T_ambient)
    
    if use_gladush:
        D_m = get_melt_depth_gladush(P_grid, v_grid, a_grid, material)
        depth_label = "Melt Pool Depth (Gladush)"
    else:
        D_m = D_rubenchik_m
        depth_label = "Melt Pool Depth (Rubenchik)"

    # 4. Convert to Microns for Plotting
    L_um = L_m * 1e6
    W_um = W_m * 1e6
    D_um = D_m * 1e6
    
    # 5. Prepare Labels and Data
    # Helper to get formatted label with units
    def get_label(var_name):
        if var_name == 'P': return "Power (W)"
        if var_name == 'v': return "Scanning Velocity (m/s)"
        if var_name == 'a': return "Spot Size (m)"
        return var_name

    x_label = get_label(x_var)
    y_label = get_label(y_var)
    
    titles = ["Melt Pool Length", "Melt Pool Width", depth_label]
    data_to_plot = [L_um, W_um, D_um]

    # 6. Plotting with Contourf and Colorbars
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))
    
    for i, ax in enumerate(axes):
        # Use contourf for filled contours, similar to the example image
        # We set 15 levels for a smooth gradient
        cf = ax.contourf(X, Y, data_to_plot[i], levels=15, cmap='inferno')
        
        # Add contour lines on top for better readability
        CS = ax.contour(X, Y, data_to_plot[i], levels=15, colors='k', linewidths=0.5)
        ax.clabel(CS, inline=True, fontsize=8, fmt='%1.0f') # Label the lines

        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        ax.set_title(titles[i], fontsize=14)
        
        # Add the colorbar with the (μm) label
        cbar = fig.colorbar(cf, ax=ax)
        cbar.set_label(r'$(\mu m)$', fontsize=12, rotation=270, labelpad=15)

    plt.tight_layout()
    plt.show()

# Deffect Criteria Plots
def plot_defect_map(x_var, y_var, x_range, y_range, fixed_params, material, 
                            layer_t=40e-6, hatch_s=100e-6, resolution=150):
    
    # 1. Setup Grid & Mapping
    x = np.linspace(x_range[0], x_range[1], resolution)
    y = np.linspace(y_range[0], y_range[1], resolution)
    X, Y = np.meshgrid(x, y)
    
    mapping = {x_var: X, y_var: Y}
    for key, val in fixed_params.items():
        mapping[key] = val
        
    P, v, a = mapping['P'], mapping['v'], mapping['a']

    # 2. Calculate Dimensions & Defect Masks
    L, W, D = rubenchik_field(P, v, a, material, T_ambient=298)
    is_balling, is_keyhole, is_lof = get_defect_masks(P, v, a, L, W, D, material, layer_t, hatch_s)
    
    # 3. Define specific region masks for plotting
    # Stable area (where no defects are True)
    stable_mask = ~is_balling & ~is_keyhole & ~is_lof
    
    # Overlap masks (where 2 or more are True)
    overlap_mask = (is_balling & is_keyhole) | (is_balling & is_lof) | (is_keyhole & is_lof)

    # 4. Visualization Setup
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Define Colors
    c_stable = 'green'
    c_keyhole = 'red' 
    c_lof = 'gold'     
    c_balling = 'gray' 
    
    # --- Layer 1: Base Colors (Solid) ---
    # We plot solid colors wherever a defect is present. 
    # Overlaps will be painted over later, but this provides the background color.
    # Using contourf with levels=[0.5, 1.5] isolates the boolean 'True' regions.
    
    # Plot Stable background
    ax.contourf(X, Y, stable_mask, levels=[0.5, 1.5], colors=[c_stable])

    # Plot solid regions for individual defects
    ax.contourf(X, Y, is_keyhole, levels=[0.5, 1.5], colors=[c_keyhole], alpha=0.6)
    ax.contourf(X, Y, is_lof, levels=[0.5, 1.5], colors=[c_lof], alpha=0.6)
    ax.contourf(X, Y, is_balling, levels=[0.5, 1.5], colors=[c_balling], alpha=0.6)
    
    # --- Layer 2: Hatches for Overlaps ---
    # We plot transparent patches with hatch patterns over the overlapping zones.
    # Hatch intensity can be increased by repeating symbols (e.g., '///' vs '/')
    
    # General Overlap (e.g., Keyhole + Balling) -> Crosshatch 'XX'
    ax.contourf(X, Y, is_keyhole & is_balling, levels=[0.5, 1.5], colors='none', hatches=['XX'])
    
    # General Overlap (e.g., LoF + others) -> Diagonal stripes '//'
    # We exclude areas that are already crosshatched to avoid messy double-hatching
    ax.contourf(X, Y, is_lof & (is_keyhole | is_balling) & ~(is_keyhole & is_balling), 
                levels=[0.5, 1.5], colors='none', hatches=['//'])

    # --- Formatting ---
    ax.set_xlabel(f"{x_var} (m/s)" if x_var == 'v' else f"{x_var} (W)", fontsize=12)
    ax.set_ylabel(f"{y_var} (W)" if y_var == 'P' else f"{y_var} (m/s)", fontsize=12)
    ax.set_title(f"L-PBF Process Map (t={layer_t*1e6:.0f}μm, h={hatch_s*1e6:.0f}μm)", fontsize=14)
    
    # Create Custom Legend
    legend_elements = [
        Patch(facecolor=c_stable, edgecolor='gray', label='Stable Window'),
        Patch(facecolor=c_keyhole, alpha=0.6, label='Keyhole Mode'),
        Patch(facecolor=c_lof, alpha=0.6, label='Lack of Fusion'),
        Patch(facecolor=c_balling, alpha=0.6, label='Balling (Unstable)'),
        Patch(facecolor='none', edgecolor='black', hatch='XX', label='Overlap: Keyhole + Balling'),
        Patch(facecolor='none', edgecolor='black', hatch='//', label='Overlap: LoF + Others'),
    ]
    ax.legend(handles=legend_elements, loc='best', frameon=True, fontsize=10)
    
    # Add thin contour lines for sharp boundaries
    ax.contour(X, Y, is_keyhole, levels=[0.5], colors='k', linewidths=0.5)
    ax.contour(X, Y, is_lof, levels=[0.5], colors='k', linewidths=0.5)
    ax.contour(X, Y, is_balling, levels=[0.5], colors='k', linewidths=0.5)

    plt.tight_layout()
    plt.show()

# 3D Gaussian Laser Beam Visualization
def gaussian_laser():
    """
    Returns the value of a 2D Gaussian function at point (x, y).
    """
    # 1. Setup the figure
    fig = plt.figure(figsize=(8, 8), dpi=100)
    ax = fig.add_subplot(111, projection='3d')

    # 2. Define Parameters
    sigma = 1.0       # Standard deviation
    qm = 2.0          # Max amplitude (arbitrary scale for visuals)
    limit = 2.5       # Plot range

    # 3. Generate Data for the Surface
    # We create a grid for the wireframe
    x = np.linspace(-limit, limit, 30)
    y = np.linspace(-limit, limit, 30)
    X, Y = np.meshgrid(x, y)
    R_squared = X**2 + Y**2
    Z = qm * np.exp(-R_squared / (2 * sigma**2))

    # 4. Plot the Main Wireframe
    # rstride/cstride control the density of the grid lines
    ax.plot_wireframe(X, Y, Z, rstride=3, cstride=3, color='black', linewidth=0.6, alpha=0.6)

    # 5. Plot the Base Ellipse (Visual Boundary)
    theta = np.linspace(0, 2*np.pi, 100)
    x_circ = limit * np.cos(theta)
    y_circ = limit * np.sin(theta)
    ax.plot(x_circ, y_circ, np.zeros_like(x_circ), color='black', linewidth=0.8)

    # 6. Plot the "Waist" Circle (at height qm * e^-1/2)
    # This represents the standard deviation radius
    z_sigma = qm * np.exp(-0.5)
    x_sigma = sigma * np.cos(theta)
    y_sigma = sigma * np.sin(theta)
    z_sigma_arr = np.full_like(x_sigma, z_sigma)

    # Plot the circle (dashed style)
    ax.plot(x_sigma, y_sigma, z_sigma_arr, color='black', linestyle='--', linewidth=1.2)

    # 7. Add Annotations and Dimension Lines

    # --- Central Axis Arrow ---
    ax.quiver(0, 0, 0, 0, 0, qm*1.2, color='black', arrow_length_ratio=0.05, linewidth=1)
    ax.text(0, 0, qm*1.25, '$q(r)$', fontsize=12, ha='center')

    # --- q_m Dimension (Right Side) ---
    # Top extension line
    ax.plot([0, 0], [0, limit], [qm, qm], color='black', linewidth=0.5) 
    # Bottom extension line (on ground)
    ax.plot([0, limit], [0, 0], [0, 0], color='black', linewidth=0.5) 
    # Vertical arrow line
    ax.quiver(limit, 0, 0, 0, 0, qm, color='black', arrow_length_ratio=0.03, linewidth=0.8)
    ax.quiver(limit, 0, qm, 0, 0, -qm, color='black', arrow_length_ratio=0.03, linewidth=0.8)
    ax.text(limit, 0, qm/2, ' $q_0$', fontsize=12, va='center')

    # --- q_m * e^-1/2 Dimension (Left Side) ---
    # Extension line from the dashed circle
    ax.plot([0, -limit], [0, 0], [z_sigma, z_sigma], color='black', linewidth=0.5)
    # Vertical arrow
    ax.quiver(-limit, 0, 0, 0, 0, z_sigma, color='black', arrow_length_ratio=0.05, linewidth=0.8)
    ax.quiver(-limit, 0, z_sigma, 0, 0, -z_sigma, color='black', arrow_length_ratio=0.05, linewidth=0.8)
    # Label
    ax.text(-limit, 0, z_sigma/2, r'$q_0 \cdot e^{-1/2}$ ', fontsize=11, ha='right', va='center')

    # --- 2*Sigma Dimension (Bottom Center) ---
    # Drop lines from the dashed circle to the ground
    ax.plot([sigma, sigma], [0, 0], [z_sigma, 0], 'k--', linewidth=0.5)
    ax.plot([-sigma, -sigma], [0, 0], [z_sigma, 0], 'k--', linewidth=0.5)

    # Horizontal dimension arrow slightly below the axis
    z_dim = -0.3 # shift down
    ax.plot([-sigma, sigma], [0, 0], [z_dim, z_dim], 'k-', linewidth=0.8)
    # Manual arrow heads for the horizontal line
    ax.text(0, 0, z_dim-0.2, r'$2\sigma$', ha='center', fontsize=12)
    # Draw ticks for the width
    ax.plot([sigma, sigma], [0, 0], [0, z_dim], 'k-', linewidth=0.5)
    ax.plot([-sigma, -sigma], [0, 0], [0, z_dim], 'k-', linewidth=0.5)


    # 8. Styling to match "Technical Drawing"
    ax.set_axis_off()          # Turn off the box
    ax.view_init(elev=20, azim=-80) # Adjust camera angle to match image
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_zlim(-0.5, qm*1.3)

    plt.show()
    return fig