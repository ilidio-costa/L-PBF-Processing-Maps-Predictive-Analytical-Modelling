from annotated_types import T
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.optimize import minimize_scalar
import numpy as np

from .physics import (
    eagar_tsai_temp,
    get_eagar_tsai_dimensions,
    rubenchik_variables,
    rubenchik_temp,
    get_rubenchik_dimensions,
    get_melt_depth_gladush_smurov,
    get_defect_masks,
)

## ======================= Eagar-Tsai PLOTS ======================= ##

def top_view_eagar_tsai(P, v, a, material, T_ambient=0, resolution=120, remove_background=False, ax=None):
    L, W, D, x_tail, x_front = get_eagar_tsai_dimensions(P, v, a, material, T_ambient=T_ambient, resolution=100)
    Tm = material['T_m']
    
    if L > 0:
        padding_x, padding_y = L * 0.2, W * 0.2
        x_min, x_max = x_tail - padding_x, x_front + padding_x
        y_min, y_max = -W/2 - padding_y, W/2 + padding_y
    else:
        x_min, x_max, y_min, y_max = -5*a, 2*a, -3*a, 3*a

    xs = np.linspace(x_min, x_max, resolution)
    ys = np.linspace(y_min, y_max, resolution)
    X, Y = np.meshgrid(xs, ys)
    
    T_field = np.zeros_like(X)
    for i in range(resolution):
        for j in range(resolution):
            T_field[i, j] = eagar_tsai_temp(X[i, j], Y[i, j], 0, P, v, a, material, T_ambient)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
        return_fig = True
    else:
        return_fig = False

    plot_data = np.ma.masked_less(T_field, Tm) if remove_background else T_field
    contour = ax.contourf(X * 1e6, Y * 1e6, plot_data, levels=50, cmap='inferno')
    if np.max(T_field) >= Tm:
        ax.contour(X * 1e6, Y * 1e6, T_field, levels=[Tm], colors='cyan', linewidths=2, linestyles='--')
    
    ax.set_aspect('equal')
    ax.set_xlabel('Length X (µm)')
    ax.set_ylabel('Width Y (µm)')
    
    dim_lbl = f"L: {L*1e6:.1f}µm | W: {W*1e6:.1f}µm" if L > 0 else "No Melt Pool"
    ax.set_title(f'Eagar-Tsai Top | P={P}W, v={v}m/s\n{dim_lbl}')
    
    if return_fig:
        plt.colorbar(contour, ax=ax, label='Temperature (K)')
        return fig
    return contour

def side_view_eagar_tsai(P, v, a, material, T_ambient=0, resolution=120, remove_background=False, ax=None):
    L, W, D, x_tail, x_front = get_eagar_tsai_dimensions(P, v, a, material, T_ambient=T_ambient,  resolution=100)
    Tm = material['T_m']
    
    if L > 0:
        padding_x, padding_z = L * 0.2, D * 0.2
        x_min, x_max = x_tail - padding_x, x_front + padding_x
        z_min, z_max = -D - padding_z, 0 
    else:
        x_min, x_max, z_min, z_max = -5*a, 2*a, -3*a, 0

    xs = np.linspace(x_min, x_max, resolution)
    zs = np.linspace(z_min, 0, resolution)
    X, Z = np.meshgrid(xs, zs)
    
    T_field = np.zeros_like(X)
    for i in range(resolution):
        for j in range(resolution):
            T_field[i, j] = eagar_tsai_temp(X[i, j], 0, Z[i, j], P, v, a, material, T_ambient)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
        return_fig = True
    else:
        return_fig = False

    plot_data = np.ma.masked_less(T_field, Tm) if remove_background else T_field
    contour = ax.contourf(X * 1e6, Z * 1e6, plot_data, levels=50, cmap='magma')
    if np.max(T_field) >= Tm:
        ax.contour(X * 1e6, Z * 1e6, T_field, levels=[Tm], colors='cyan', linewidths=2, linestyles='--')
    
    ax.set_aspect('equal')
    ax.set_xlabel('Length X (µm)')
    ax.set_ylabel('Depth Z (µm)')
    
    dim_lbl = f"L: {L*1e6:.1f}µm | D: {D*1e6:.1f}µm" if L > 0 else "No Melt Pool"
    ax.set_title(f'Eagar-Tsai Side | P={P}W, v={v}m/s\n{dim_lbl}')
    
    if return_fig:
        plt.colorbar(contour, ax=ax, label='Temperature (K)')
        return fig
    return contour

def plot_process_et_grid_views(P_range, v_range, a, material, T_ambient=0, resolution=100, remove_background=False):
    """
    Generates process window grid plots with ROBUST LIMIT DETECTION.
    - Pre-scans ALL combinations to find the exact global bounding box.
    - Ensures no clipping and minimal whitespace.
    - Dynamic figure sizing based on aspect ratio.
    - Optional background removal to show only the melt pool.
    """
    Tm = material['T_m']
    n_P = len(P_range)
    n_v = len(v_range)
    
    print(f"[GRID] Pre-scanning {n_P * n_v} combinations for exact limits...")

    # --- 1. ROBUST LIMIT SEARCH (Scan All) ---
    global_min_x = -a 
    global_max_x = a
    global_max_W = a
    global_max_D = a
    global_peak_T = Tm

    for P in P_range:
        for v in v_range:
            L, W, D, x_tail, x_front = get_eagar_tsai_dimensions(P, v, a, material, T_ambient, resolution=40)
            
            if L > 0:
                global_min_x = min(global_min_x, x_tail)
                global_max_x = max(global_max_x, x_front)
                global_max_W = max(global_max_W, W)
                global_max_D = max(global_max_D, D)
                
                res_peak = minimize_scalar(
                    lambda x: - eagar_tsai_temp(x, 0, 0, P, v, a, material, T_ambient), 
                    bounds=(-5*a, a), method='bounded'
                )
                global_peak_T = max(global_peak_T, -res_peak.fun)

    # --- 2. DEFINE LIMITS & PADDING ---
    span_x = global_max_x - global_min_x
    padding_x = span_x * 0.1
    x_min = global_min_x - padding_x
    x_max = global_max_x + padding_x
    
    y_span = global_max_W
    padding_y = y_span * 0.1
    y_min = -(global_max_W/2) - padding_y
    y_max = (global_max_W/2) + padding_y
    
    z_min = -global_max_D * 1.15
    z_max = 0
    
    tlims = (300, global_peak_T * 1.05)
    
    # 50 fixed temperature levels globally
    global_levels = np.linspace(tlims[0], tlims[1], 50)

    # --- 3. DYNAMIC ASPECT RATIO & FIGURE SIZE ---
    dx = x_max - x_min
    dy = y_max - y_min
    dz = abs(z_min)
    
    ratio_top = dy / dx
    ratio_side = dz / dx
    
    subplot_w = 3.5
    fig_w = n_P * subplot_w
    
    # [FIX]: Guarantee at least 0.85 inches of height per row for BOTH views
    min_row_height = 0.85 
    fig_h_top = max((n_v * subplot_w * ratio_top), n_v * min_row_height) + 1.5 
    fig_h_side = max((n_v * subplot_w * ratio_side), n_v * min_row_height) + 1.5

    print(f"[GRID] Limits: X[{x_min*1e6:.0f}:{x_max*1e6:.0f}]µm | W[{global_max_W*1e6:.0f}]µm")
    print(f"[GRID] Layout: Top Aspect {ratio_top:.2f}, Side Aspect {ratio_side:.2f}")

    # --- 4. PLOTTING SETUP ---
    X_grid = np.linspace(x_min, x_max, resolution)
    Y_grid = np.linspace(y_min, y_max, resolution)
    Z_grid = np.linspace(z_min, z_max, resolution)
    
    X_mesh_top, Y_mesh_top = np.meshgrid(X_grid, Y_grid)
    X_mesh_side, Z_mesh_side = np.meshgrid(X_grid, Z_grid)

    # ================= TOP VIEW =================
    fig_top, axes_top = plt.subplots(nrows=n_v, ncols=n_P, 
                                     figsize=(fig_w, fig_h_top), 
                                     sharex=True, sharey=True,
                                     constrained_layout=True)
    
    if n_v == 1 and n_P == 1: axes_top = np.array([[axes_top]])
    elif n_v == 1: axes_top = axes_top.reshape(1, -1)
    elif n_P == 1: axes_top = axes_top.reshape(-1, 1)

    fig_top.suptitle(f"{material['name']} Melt Pool Top Views Process Map at {T_ambient}K", fontsize=16)

    for i, v in enumerate(v_range):
        for j, P in enumerate(P_range):
            ax = axes_top[i, j]
            
            T_field = np.zeros_like(X_mesh_top)
            for ri in range(resolution):
                for ci in range(resolution):
                    T_field[ri, ci] = eagar_tsai_temp(X_mesh_top[ri, ci], Y_mesh_top[ri, ci], 0, P, v, a, material, T_ambient)
            
            plot_data = T_field
            if remove_background:
                plot_data = np.ma.masked_less(T_field, Tm)

            ax.contourf(X_mesh_top*1e6, Y_mesh_top*1e6, plot_data, 
                        levels=global_levels, cmap='inferno', 
                        vmin=tlims[0], vmax=tlims[1], extend='both')
            
            ax.contour(X_mesh_top*1e6, Y_mesh_top*1e6, T_field, 
                       levels=[Tm], colors='cyan', linewidths=1.5, linestyles='--')
            
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.2)
            
            if i == n_v - 1: ax.set_xlabel("X (µm)")
            if j == 0: ax.set_ylabel(f"v={v*1000:.0f} mm/s\nY (µm)", labelpad=10)
            if i == 0: ax.set_title(f"P={P:.0f}W")

    sm_top = plt.cm.ScalarMappable(cmap='inferno', norm=plt.Normalize(vmin=tlims[0], vmax=tlims[1]))
    cbar = fig_top.colorbar(sm_top, ax=axes_top, location='right', aspect=30)
    cbar.set_label('Temperature (K)', fontsize=12)

    # ================= SIDE VIEW =================
    fig_side, axes_side = plt.subplots(nrows=n_v, ncols=n_P, 
                                       figsize=(fig_w, fig_h_side), 
                                       sharex=True, sharey=True,
                                       constrained_layout=True)
    
    if n_v == 1 and n_P == 1: axes_side = np.array([[axes_side]])
    elif n_v == 1: axes_side = axes_side.reshape(1, -1)
    elif n_P == 1: axes_side = axes_side.reshape(-1, 1)

    fig_side.suptitle(f"{material['name']} Melt Pool Side Views Process Map at {T_ambient}K", fontsize=16)

    for i, v in enumerate(v_range):
        for j, P in enumerate(P_range):
            ax = axes_side[i, j]
            
            T_field = np.zeros_like(X_mesh_side)
            for ri in range(resolution):
                for ci in range(resolution):
                    T_field[ri, ci] = eagar_tsai_temp(X_mesh_side[ri, ci], 0, Z_mesh_side[ri, ci], P, v, a, material, T_ambient)
            
            plot_data = T_field
            if remove_background:
                plot_data = np.ma.masked_less(T_field, Tm)

            ax.contourf(X_mesh_side*1e6, Z_mesh_side*1e6, plot_data, 
                        levels=global_levels, cmap='inferno', 
                        vmin=tlims[0], vmax=tlims[1], extend='both')
            
            ax.contour(X_mesh_side*1e6, Z_mesh_side*1e6, T_field, 
                       levels=[Tm], colors='cyan', linewidths=1.5, linestyles='--')
            
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.2)
            
            if i == n_v - 1: ax.set_xlabel("X (µm)")
            if j == 0: ax.set_ylabel(f"v={v*1000:.0f} mm/s\nZ (µm)", labelpad=10)
            if i == 0: ax.set_title(f"P={P:.0f}W")

    sm_side = plt.cm.ScalarMappable(cmap='inferno', norm=plt.Normalize(vmin=tlims[0], vmax=tlims[1]))
    cbar_side = fig_side.colorbar(sm_side, ax=axes_side, location='right', aspect=30)
    cbar_side.set_label('Temperature (K)', fontsize=12)

    return fig_top, fig_side

## ======================= RUBENCHIK PLOTS ======================= ##

def top_view_rubenchik(P, v, a, material, T_ambient=0, resolution=120, remove_background=False, ax=None):
    L, W, D, x_tail, x_front = get_rubenchik_dimensions(P, v, a, material, T_ambient, resolution=100)
    Tm = material['T_m']
    
    if L > 0:
        padding_x, padding_y = L * 0.2, W * 0.2
        x_min, x_max = x_tail - padding_x, x_front + padding_x
        y_min, y_max = -W/2 - padding_y, W/2 + padding_y
    else:
        x_min, x_max, y_min, y_max = -5*a, 2*a, -3*a, 3*a

    xs = np.linspace(x_min, x_max, resolution)
    ys = np.linspace(y_min, y_max, resolution)
    X, Y = np.meshgrid(xs, ys)
    
    T_field = np.zeros_like(X)
    for i in range(resolution):
        for j in range(resolution):
            coords, B, p = rubenchik_variables(X[i, j], Y[i, j], 0, material, P, v, a, T_ambient)
            T_field[i, j] = rubenchik_temp(coords, B, p, material)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
        return_fig = True
    else:
        return_fig = False

    plot_data = np.ma.masked_less(T_field, Tm) if remove_background else T_field
    contour = ax.contourf(X * 1e6, Y * 1e6, plot_data, levels=50, cmap='inferno')
    if np.max(T_field) >= Tm:
        ax.contour(X * 1e6, Y * 1e6, T_field, levels=[Tm], colors='cyan', linewidths=2, linestyles='--')
    
    ax.set_aspect('equal')
    ax.set_xlabel('Length X (µm)')
    ax.set_ylabel('Width Y (µm)')
    
    dim_lbl = f"L: {L*1e6:.1f}µm | W: {W*1e6:.1f}µm" if L > 0 else "No Melt Pool"
    ax.set_title(f'Rubenchik Top | P={P}W, v={v}m/s\n{dim_lbl}')
    
    if return_fig:
        plt.colorbar(contour, ax=ax, label='Temperature (K)')
        return fig
    return contour

def side_view_rubenchik(P, v, a, material, T_ambient=0, resolution=120, remove_background=False, ax=None):
    L, W, D, x_tail, x_front = get_rubenchik_dimensions(P, v, a, material, T_ambient, resolution=100)
    Tm = material['T_m']
    
    if L > 0:
        padding_x, padding_z = L * 0.2, D * 0.2
        x_min, x_max = x_tail - padding_x, x_front + padding_x
        z_min, z_max = -D - padding_z, 0 
    else:
        x_min, x_max, z_min, z_max = -5*a, 2*a, -3*a, 0

    xs = np.linspace(x_min, x_max, resolution)
    zs = np.linspace(z_min, 0, resolution)
    X, Z = np.meshgrid(xs, zs)
    
    T_field = np.zeros_like(X)
    for i in range(resolution):
        for j in range(resolution):
            coords, B, p = rubenchik_variables(X[i, j], 0, Z[i, j], material, P, v, a, T_ambient)
            T_field[i, j] = rubenchik_temp(coords, B, p, material)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
        return_fig = True
    else:
        return_fig = False

    plot_data = np.ma.masked_less(T_field, Tm) if remove_background else T_field
    contour = ax.contourf(X * 1e6, Z * 1e6, plot_data, levels=50, cmap='magma')
    if np.max(T_field) >= Tm:
        ax.contour(X * 1e6, Z * 1e6, T_field, levels=[Tm], colors='cyan', linewidths=2, linestyles='--')
    
    ax.set_aspect('equal')
    ax.set_xlabel('Length X (µm)')
    ax.set_ylabel('Depth Z (µm)')
    
    dim_lbl = f"L: {L*1e6:.1f}µm | D: {D*1e6:.1f}µm" if L > 0 else "No Melt Pool"
    ax.set_title(f'Rubenchik Side | P={P}W, v={v}m/s\n{dim_lbl}')
    
    if return_fig:
        plt.colorbar(contour, ax=ax, label='Temperature (K)')
        return fig
    return contour

def plot_process_r_grid_views(P_range, v_range, a, material, T_ambient=0, resolution=100, remove_background=False):
    """
    Generates process window grid plots with ROBUST LIMIT DETECTION.
    - Pre-scans ALL combinations to find the exact global bounding box.
    - Ensures no clipping and minimal whitespace.
    - Dynamic figure sizing based on aspect ratio.
    - Optional background removal to show only the melt pool.
    """
    Tm = material['T_m']
    n_P = len(P_range)
    n_v = len(v_range)
    
    print(f"[GRID] Pre-scanning {n_P * n_v} combinations for exact limits...")

    # --- 1. ROBUST LIMIT SEARCH (Scan All) ---
    global_min_x = -a 
    global_max_x = a
    global_max_W = a
    global_max_D = a
    global_peak_T = Tm

    for P in P_range:
        for v in v_range:
            L, W, D, x_tail, x_front = get_rubenchik_dimensions(P, v, a, material, T_ambient, resolution=40)
            
            if L > 0:
                global_min_x = min(global_min_x, x_tail)
                global_max_x = max(global_max_x, x_front)
                global_max_W = max(global_max_W, W)
                global_max_D = max(global_max_D, D)

                coords, B_val, p_val = rubenchik_variables(0, 0, 0, material, P, v, a, T_ambient)  # Ensure variables are computed for peak search

                res_peak = minimize_scalar(
                    lambda x: - rubenchik_temp((x, 0, 0), B_val, p_val, material), 
                    bounds=(-5*a, a), method='bounded'
                )
                global_peak_T = max(global_peak_T, -res_peak.fun)

    # --- 2. DEFINE LIMITS & PADDING ---
    span_x = global_max_x - global_min_x
    padding_x = span_x * 0.1
    x_min = global_min_x - padding_x
    x_max = global_max_x + padding_x
    
    y_span = global_max_W
    padding_y = y_span * 0.1
    y_min = -(global_max_W/2) - padding_y
    y_max = (global_max_W/2) + padding_y
    
    z_min = -global_max_D * 1.15
    z_max = 0
    
    tlims = (300, global_peak_T * 1.05)
    
    # 50 fixed temperature levels globally
    global_levels = np.linspace(tlims[0], tlims[1], 50)

    # --- 3. DYNAMIC ASPECT RATIO & FIGURE SIZE ---
    dx = x_max - x_min
    dy = y_max - y_min
    dz = abs(z_min)
    
    ratio_top = dy / dx
    ratio_side = dz / dx
    
    subplot_w = 3.5
    fig_w = n_P * subplot_w
    
    # [FIX]: Guarantee at least 0.85 inches of height per row for BOTH views
    min_row_height = 0.85 
    fig_h_top = max((n_v * subplot_w * ratio_top), n_v * min_row_height) + 1.5 
    fig_h_side = max((n_v * subplot_w * ratio_side), n_v * min_row_height) + 1.5

    print(f"[GRID] Limits: X[{x_min*1e6:.0f}:{x_max*1e6:.0f}]µm | W[{global_max_W*1e6:.0f}]µm")
    print(f"[GRID] Layout: Top Aspect {ratio_top:.2f}, Side Aspect {ratio_side:.2f}")

    # --- 4. PLOTTING SETUP ---
    X_grid = np.linspace(x_min, x_max, resolution)
    Y_grid = np.linspace(y_min, y_max, resolution)
    Z_grid = np.linspace(z_min, z_max, resolution)
    
    X_mesh_top, Y_mesh_top = np.meshgrid(X_grid, Y_grid)
    X_mesh_side, Z_mesh_side = np.meshgrid(X_grid, Z_grid)

    # ================= TOP VIEW =================
    fig_top, axes_top = plt.subplots(nrows=n_v, ncols=n_P, 
                                     figsize=(fig_w, fig_h_top), 
                                     sharex=True, sharey=True,
                                     constrained_layout=True)
    
    if n_v == 1 and n_P == 1: axes_top = np.array([[axes_top]])
    elif n_v == 1: axes_top = axes_top.reshape(1, -1)
    elif n_P == 1: axes_top = axes_top.reshape(-1, 1)

    fig_top.suptitle(f"{material['name']} Melt Pool Top Views Process Map at {T_ambient}K", fontsize=16)

    for i, v in enumerate(v_range):
        for j, P in enumerate(P_range):
            ax = axes_top[i, j]
            
            T_field = np.zeros_like(X_mesh_top)
            for ri in range(resolution):
                for ci in range(resolution):
                    coords, B, p = rubenchik_variables(X_mesh_top[ri, ci], Y_mesh_top[ri, ci], 0, material, P, v, a, T_ambient)
                    T_field[ri, ci] = rubenchik_temp(coords, B, p, material)

            plot_data = T_field
            if remove_background:
                plot_data = np.ma.masked_less(T_field, Tm)

            ax.contourf(X_mesh_top*1e6, Y_mesh_top*1e6, plot_data, 
                        levels=global_levels, cmap='inferno', 
                        vmin=tlims[0], vmax=tlims[1], extend='both')
            
            ax.contour(X_mesh_top*1e6, Y_mesh_top*1e6, T_field, 
                       levels=[Tm], colors='cyan', linewidths=1.5, linestyles='--')
            
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.2)
            
            if i == n_v - 1: ax.set_xlabel("X (µm)")
            if j == 0: ax.set_ylabel(f"v={v*1000:.0f} mm/s\nY (µm)", labelpad=10)
            if i == 0: ax.set_title(f"P={P:.0f}W")

    sm_top = plt.cm.ScalarMappable(cmap='inferno', norm=plt.Normalize(vmin=tlims[0], vmax=tlims[1]))
    cbar = fig_top.colorbar(sm_top, ax=axes_top, location='right', aspect=30)
    cbar.set_label('Temperature (K)', fontsize=12)

    # ================= SIDE VIEW =================
    fig_side, axes_side = plt.subplots(nrows=n_v, ncols=n_P, 
                                       figsize=(fig_w, fig_h_side), 
                                       sharex=True, sharey=True,
                                       constrained_layout=True)
    
    if n_v == 1 and n_P == 1: axes_side = np.array([[axes_side]])
    elif n_v == 1: axes_side = axes_side.reshape(1, -1)
    elif n_P == 1: axes_side = axes_side.reshape(-1, 1)

    fig_side.suptitle(f"{material['name']} Melt Pool Side Views Process Map at {T_ambient}K", fontsize=16)

    for i, v in enumerate(v_range):
        for j, P in enumerate(P_range):
            ax = axes_side[i, j]
            
            T_field = np.zeros_like(X_mesh_side)
            for ri in range(resolution):
                for ci in range(resolution):
                    coords, B, p = rubenchik_variables(X_mesh_top[ri, ci], 0, Z_mesh_side[ri, ci], material, P, v, a, T_ambient)
                    T_field[ri, ci] = rubenchik_temp(coords, B, p, material)

            plot_data = T_field
            if remove_background:
                plot_data = np.ma.masked_less(T_field, Tm)

            ax.contourf(X_mesh_side*1e6, Z_mesh_side*1e6, plot_data, 
                        levels=global_levels, cmap='inferno', 
                        vmin=tlims[0], vmax=tlims[1], extend='both')
            
            ax.contour(X_mesh_side*1e6, Z_mesh_side*1e6, T_field, 
                       levels=[Tm], colors='cyan', linewidths=1.5, linestyles='--')
            
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.2)
            
            if i == n_v - 1: ax.set_xlabel("X (µm)")
            if j == 0: ax.set_ylabel(f"v={v*1000:.0f} mm/s\nZ (µm)", labelpad=10)
            if i == 0: ax.set_title(f"P={P:.0f}W")

    sm_side = plt.cm.ScalarMappable(cmap='inferno', norm=plt.Normalize(vmin=tlims[0], vmax=tlims[1]))
    cbar_side = fig_side.colorbar(sm_side, ax=axes_side, location='right', aspect=30)
    cbar_side.set_label('Temperature (K)', fontsize=12)

    return fig_top, fig_side

## ============== Processing Map Melt Pool Dimension Plotting ================= ##

def plot_melt_pool_dimensions(x_var, y_var, x_range, y_range, fixed_params, material, T_ambient=298, use_rubenchik=True, use_gladush=True, resolution=40):
    """
    Plots formatted contour maps for Melt Pool Length, Width, and Depth.
    Iterates over a grid of parameters and calculates dimensions using the chosen analytical models.
    
    Parameters:
    -----------
    x_var : str
        The variable to plot on the X-axis. Options are 'P' (Power), 'v' (Velocity), or 'a' (Spot Size).
    y_var : str
        The variable to plot on the Y-axis. Options are 'P' (Power), 'v' (Velocity), or 'a' (Spot Size).
    x_range : tuple
        The (min, max) limits for the x_var axis.
    y_range : tuple
        The (min, max) limits for the y_var axis.
    fixed_params : dict
        A dictionary containing the fixed value for the parameter not being varied.
        For example, if x_var='v' and y_var='P', fixed_params must be {'a': 50e-6}.
    material : dict
        Dictionary containing the thermophysical properties of the material.
    T_ambient : float, optional
        Ambient temperature in Kelvin. Default is 298.
    use_rubenchik : bool, optional
        If True, uses the Rubenchik model for Length and Width. If False, uses Eagar-Tsai. Default is True.
    use_gladush : bool, optional
        If True, uses the Gladush-Smurov model for Depth. If False, uses the base model chosen above. Default is True.
    resolution : int, optional
        The number of points to calculate along each axis. Higher means smoother plots but slower calculation. Default is 40.
    """
    
    # 1. Setup Grid
    x = np.linspace(x_range[0], x_range[1], resolution)
    y = np.linspace(y_range[0], y_range[1], resolution)
    X, Y = np.meshgrid(x, y)
    
    # 2. Initialize Dimension Grids
    L_grid = np.zeros_like(X)
    W_grid = np.zeros_like(X)
    D_grid = np.zeros_like(X)
    
    print(f"Calculating {resolution}x{resolution} processing map. This may take a moment...")

    # 3. Calculate Dimensions Iteratively
    for i in range(resolution):
        for j in range(resolution):
            # Map current coordinates to P, v, a
            current_params = fixed_params.copy()
            current_params[x_var] = X[i, j]
            current_params[y_var] = Y[i, j]
            
            P = current_params.get('P')
            v = current_params.get('v')
            a = current_params.get('a')
            
            # Select Base Model (Length and Width)
            if use_rubenchik:
                L, W, D, _, _ = get_rubenchik_dimensions(P, v, a, material, T_ambient, resolution=50)
            else:
                L, W, D, _, _ = get_eagar_tsai_dimensions(P, v, a, material, T_ambient, resolution=50)
            
            # Override Depth with Gladush-Smurov if requested
            if use_gladush:
                D = get_melt_depth_gladush_smurov(P, v, a, material)
                
            # Store in grids (converted to microns immediately)
            L_grid[i, j] = L * 1e6
            W_grid[i, j] = W * 1e6
            D_grid[i, j] = D * 1e6

    # 4. Prepare Labels and Titles
    def get_label(var_name):
        if var_name == 'P': return "Power (W)"
        if var_name == 'v': return "Scanning Velocity (m/s)"
        if var_name == 'a': return "Spot Size Radius (m)"
        return var_name

    x_label = get_label(x_var)
    y_label = get_label(y_var)
    
    model_name = "Rubenchik" if use_rubenchik else "Eagar-Tsai"
    depth_name = "Gladush-Smurov" if use_gladush else model_name

    titles = [f"Length ({model_name})", f"Width ({model_name})", f"Depth ({depth_name})"]
    data_to_plot = [L_grid, W_grid, D_grid]

    # 5. Plotting with Contourf and Colorbars
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    
    for idx, ax in enumerate(axes):
        # Base filled contour using 'inferno'
        cf = ax.contourf(X, Y, data_to_plot[idx], levels=20, cmap='inferno')
        
        # Overlay with thin contour lines for readability
        CS = ax.contour(X, Y, data_to_plot[idx], levels=10, colors='k', linewidths=0.5, alpha=0.6)
        ax.clabel(CS, inline=True, fontsize=8, fmt='%1.0f') 

        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        ax.set_title(titles[idx], fontsize=14)
        
        # Add the colorbar with the μm label
        cbar = fig.colorbar(cf, ax=ax)
        cbar.set_label(r'$(\mu m)$', fontsize=12, rotation=270, labelpad=15)

    plt.tight_layout()
    return fig

## ==================== Deffect Criteria Plots ======================= ##

def plot_defect_map(x_var, y_var, x_range, y_range, fixed_params, material, layer_t=40e-6, hatch_s=100e-6, resolution=150):
    
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
    return fig









## ======================= Random PLOTS ======================= ##
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