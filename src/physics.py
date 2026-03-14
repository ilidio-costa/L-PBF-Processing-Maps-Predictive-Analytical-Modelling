import numpy as np
from scipy.integrate import fixed_quad
from scipy.optimize import minimize_scalar, brentq
import importlib

## ======================= Eagar-Tsai Model ======================= ##

def s_func_substituted(u, x, y, z, v, alpha, a):
    """ Transformed Integrand: s = u^2 (Removes singularity) """
    u = np.atleast_1d(u)
    s = u**2
    denom = 4 * alpha * s + a**2
    term_z = z**2 / (4 * alpha * s) if z != 0 else np.zeros_like(s)
    term_lat = (y**2 + (x + v * s)**2) / denom
    return 2 * np.exp(-term_z - term_lat) / denom

def eagar_tsai_temp(x, y, z, P, v, a, material, T_ambient=0):
    """
    Calculates T using the optimized u-substitution method 
    with high-speed dynamic spatial boundaries (Fast Fixed Quad).
    """
    A_val = material['A']
    k = material['k']
    alpha = material['alpha']
    
    pre_factor = (A_val * P / (np.pi * k)) * np.sqrt(alpha / np.pi)

    # 1. Locate Theoretical Peak
    t_peak = max(0.0, -x / v) + (a / v)
    u_peak_guess = np.sqrt(t_peak)
    max_val = float(np.squeeze(s_func_substituted(u_peak_guess, x, y, z, v, alpha, a)))
    
    # Underflow check: if the peak is virtually zero, return ambient
    if max_val < 1e-20:
        return T_ambient
        
    # 2. Dynamic Integration Bounds (8-Sigma Window)
    time_spread = np.sqrt(4.0 * alpha * t_peak) / v
    time_spread = max(time_spread, a / v)  # Fallback to avoid overly narrow bounds
    
    t_lower = max(0.0, t_peak - 8.0 * time_spread)
    t_upper = t_peak + 8.0 * time_spread
    
    u_lower = np.sqrt(t_lower)
    u_upper = np.sqrt(t_upper)

    # 3. Integrate Area
    integral_val, _ = fixed_quad(
        s_func_substituted, u_lower, u_upper, 
        args=(x, y, z, v, alpha, a), n=250  
    )
    
    return (pre_factor * integral_val) + T_ambient

def get_eagar_tsai_dimensions(P, v, a, material, T_ambient=0, resolution=100):
    """
    Calculates Melt Pool Dimensions by performing a global search for 
    max width and max depth along the melt pool length.
    """
    Tm = material['T_m']
    tol = a / float(resolution) if resolution else 1e-6 

    res_peak = minimize_scalar(
        lambda x: -eagar_tsai_temp(x, 0, 0, P, v, a, material, T_ambient), 
        bounds=(-5*a, a), method='bounded', options={'xatol': tol}
    )
    x_peak, T_max = res_peak.x, -res_peak.fun
    
    if T_max < Tm:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    func_x = lambda x: eagar_tsai_temp(x, 0, 0, P, v, a, material, T_ambient) - Tm
    
    step = a
    x_scan_fwd = x_peak + step
    while eagar_tsai_temp(x_scan_fwd, 0, 0, P, v, a, material, T_ambient) > Tm:
        x_scan_fwd += step
        step *= 1.5
        if x_scan_fwd > x_peak + 0.1: break 
    try: x_front = brentq(func_x, x_peak, x_scan_fwd, xtol=tol)
    except ValueError: x_front = x_peak

    step = a
    x_scan_bwd = x_peak - step
    while eagar_tsai_temp(x_scan_bwd, 0, 0, P, v, a, material, T_ambient) > Tm:
        x_scan_bwd -= step
        step *= 1.5
        if x_scan_bwd < x_peak - 0.1: break
    try: x_tail = brentq(func_x, x_scan_bwd, x_peak, xtol=tol)
    except ValueError: x_tail = x_peak
    
    length = x_front - x_tail

    def get_depth_at_x(x_loc):
        if eagar_tsai_temp(x_loc, 0, 0, P, v, a, material, T_ambient) < Tm: return 0.0
        func_z = lambda z: eagar_tsai_temp(x_loc, 0, z, P, v, a, material, T_ambient) - Tm
        z_scan, step_z = -a, a
        while eagar_tsai_temp(x_loc, 0, z_scan, P, v, a, material, T_ambient) > Tm:
            z_scan -= step_z
            step_z *= 1.5
            if z_scan < -0.01: break
        try: return abs(brentq(func_z, z_scan, 0, xtol=tol))
        except ValueError: return 0.0

    res_d = minimize_scalar(
        lambda x: -get_depth_at_x(x), bounds=(x_tail, x_front), 
        method='bounded', options={'xatol': tol}
    )
    depth = -res_d.fun

    def get_width_at_x(x_loc):
        if eagar_tsai_temp(x_loc, 0, 0, P, v, a, material, T_ambient) < Tm: return 0.0
        func_y = lambda y: eagar_tsai_temp(x_loc, y, 0, P, v, a, material, T_ambient) - Tm
        y_scan, step_y = a, a
        while eagar_tsai_temp(x_loc, y_scan, 0, P, v, a, material, T_ambient) > Tm:
            y_scan += step_y
            step_y *= 1.5
            if y_scan > 0.01: break
        try: return brentq(func_y, 0, y_scan, xtol=tol) * 2.0
        except ValueError: return 0.0

    res_w = minimize_scalar(
        lambda x: -get_width_at_x(x), bounds=(x_tail, x_front), 
        method='bounded', options={'xatol': tol} 
    )
    width = -res_w.fun

    return length, width, depth, x_tail, x_front

## ======================= Rubenchik Model ====================== ##

def g_func_substituted(u, xi, yi, zi, p):
    """ Transformed Integrand for the dimensionless temperature function g. """
    u = np.atleast_1d(u)
    tau = u**2
    denom_geom = 4 * p * tau + 1
    term_z = (zi**2) / (4 * tau) if zi != 0 else np.zeros_like(tau)
    term_lateral = (yi**2 + (xi + tau)**2) / denom_geom
    return 2 * np.exp(-term_z - term_lateral) / denom_geom

def rubenchik_variables(x, y, z, material, P, v, a, T_ambient=0):
    """ Computes the dimensionless variables xi, yi, zi for Rubenchik's model. """
    rho, Cp, Tm = material['rho'], material['C_p'], material['T_m']
    alpha, A_val = material['alpha'], material['A']
    
    p_val = alpha / (v * a)
    xi = x / a
    yi = y / a
    zi = z / (np.sqrt( alpha * a / v )) 

    P_eff = Tm/(Tm - T_ambient) * P

    numerator = A_val * P_eff
    denominator = np.pi * rho * Cp * Tm * np.sqrt(alpha * v * a**3)
    B_val = numerator / denominator

    return (xi, yi, zi), B_val, p_val

def rubenchik_temp(coords, B, p, material):
    """ Computes the temperature in Kelvin using the optimized Fast Fixed Quad with dynamic bounds. """
    xi, yi, zi = coords
    
    # 1. Locate Theoretical Peak in dimensionless time (tau)
    tau_peak = max(0.0, -xi) + 1.0
    u_peak_guess = np.sqrt(tau_peak)
    max_val = float(np.squeeze(g_func_substituted(u_peak_guess, xi, yi, zi, p)))

    # Underflow check
    if max_val < 1e-20:
        return 0.0

    # 2. Dynamic Integration Bounds in dimensionless time
    tau_spread = np.sqrt(4.0 * p * tau_peak)
    tau_spread = max(tau_spread, 1.0) # Fallback
    
    tau_lower = max(0.0, tau_peak - 8.0 * tau_spread)
    tau_upper = tau_peak + 8.0 * tau_spread
    
    u_lower = np.sqrt(tau_lower)
    u_upper = np.sqrt(tau_upper)

    # 3. Integrate Area
    integral_val, _ = fixed_quad(
        g_func_substituted, u_lower, u_upper, 
        args=(xi, yi, zi, p), n=250
    )

    T_rise_kelvin = material['T_m'] * (B / np.sqrt(np.pi)) * integral_val
    return T_rise_kelvin

def get_rubenchik_dimensions(P, v, a, material, T_ambient=0, resolution=100):
    """ Calculates Melt Pool Dimensions using the dimensionless formulation. """
    Tm = material['T_m']
    tol = a / float(resolution) if resolution else 1e-6 

    def temp_at(x, y, z):
        coords, B, p = rubenchik_variables(x, y, z, material, P, v, a, T_ambient)
        return rubenchik_temp(coords, B, p, material)

    res_peak = minimize_scalar(
        lambda x: - temp_at(x, 0, 0), bounds=(-5*a, a), 
        method='bounded', options={'xatol': tol}
    )
    x_peak, T_max = res_peak.x, -res_peak.fun
    if T_max < Tm: return 0.0, 0.0, 0.0, 0.0, 0.0

    func_x = lambda x: temp_at(x, 0, 0) - Tm
    step = a
    x_scan_fwd = x_peak + step
    while temp_at(x_scan_fwd, 0, 0) > Tm:
        x_scan_fwd += step
        step *= 1.5
        if x_scan_fwd > x_peak + 0.1: break
    try: x_front = brentq(func_x, x_peak, x_scan_fwd, xtol=tol)
    except ValueError: x_front = x_peak

    step = a
    x_scan_bwd = x_peak - step
    while temp_at(x_scan_bwd, 0, 0) > Tm:
        x_scan_bwd -= step
        step *= 1.5
        if x_scan_bwd < x_peak - 0.1: break
    try: x_tail = brentq(func_x, x_scan_bwd, x_peak, xtol=tol)
    except ValueError: x_tail = x_peak
    length = x_front - x_tail

    def get_depth_at_x(x_loc):
        if temp_at(x_loc, 0, 0) < Tm: return 0.0
        func_z = lambda z: temp_at(x_loc, 0, z) - Tm
        z_scan, step_z = -a, a
        while temp_at(x_loc, 0, z_scan) > Tm:
            z_scan -= step_z
            step_z *= 1.5
            if z_scan < -0.01: break
        try: return abs(brentq(func_z, z_scan, 0, xtol=tol))
        except ValueError: return 0.0

    res_d = minimize_scalar(
        lambda x: -get_depth_at_x(x), bounds=(x_tail, x_front), 
        method='bounded', options={'xatol': tol}
    )
    depth = -res_d.fun

    def get_width_at_x(x_loc):
        if temp_at(x_loc, 0, 0) < Tm: return 0.0
        func_y = lambda y: temp_at(x_loc, y, 0) - Tm
        y_scan, step_y = a, a
        while temp_at(x_loc, y_scan, 0) > Tm:
            y_scan += step_y
            step_y *= 1.5
            if y_scan > 0.01: break
        try: return brentq(func_y, 0, y_scan, xtol=tol) * 2.0
        except ValueError: return 0.0

    res_w = minimize_scalar(
        lambda x: -get_width_at_x(x), bounds=(x_tail, x_front), 
        method='bounded', options={'xatol': tol} 
    )
    width = -res_w.fun

    return length, width, depth, x_tail, x_front

def rubenchik_interpolated_dimensions(P, v, a, material, T_ambient=0):
    """
    Calculates Melt Pool Dimensions using the algebraic interpolations 
    from Rubenchik's 2018 paper.
    """
    
    # Fetch your exact dimensionless variables from physics.py
    # rubenchik_variables returns: (xi, yi, zi), B_val, p_val
    coords , B, p = rubenchik_variables(0, 0, 0, material, P, v, a, T_ambient)
    
    # =====================================================================
    # ⚠️ EQUATIONS FROM YOUR UPLOADED IMAGE GO HERE
    # =====================================================================
    # Plug in the exact coefficients from the screenshot. 
    # They generally follow a power-law format, something like:
    
    dimensionless_depth = (a / np.sqrt(p)) * (
        0.008 - 0.0048 * B - 0.047 * p - 0.099 * B * p 
        + (0.32 + 0.015 * B) * p * np.log(p) 
        + np.log(B) * (0.0056 - 0.89 * p + 0.29 * p * np.log(p))
    )

    dimensionless_width = (a / (B * p**3)) * (
        0.0021 - 0.047 * p + 0.34 * p**2 - 1.9 * p**3 - 0.33 * p**4
        + B * (0.00066 - 0.0070 * p - 0.00059 * p**2 + 2.8 * p**3 - 0.12 * p**4)
        + B**2 * (-0.00070 + 0.015 * p - 0.12 * p**2 + 0.59 * p**3 - 0.023 * p**4)
        + B**3 * (0.00001 - 0.00022 * p + 0.0020 * p**2 - 0.0085 * p**3 + 0.0014 * p**4)
    )

    dimensionless_length = (a / p**2) * (
        0.0053 - 0.21 * p + 1.3 * p**2 + (-0.11 - 0.17 * B) * p**2 * np.log(p)
        + B * (-0.0062 + 0.23 * p + 0.75 * p**2)
    )
    # =====================================================================
    
    # Convert back to dimensional units (meters)
    # Note: Check the paper to ensure width isn't returning the half-width!
    alpha = material['alpha']
    depth = dimensionless_depth * np.sqrt(alpha * a / v)
    width = dimensionless_width * a 
    length = dimensionless_length * a

    return length, width, depth

## ======================= Gladush-Smurov Model ======================= ##
def get_melt_depth_gladush_smurov(P, v, a, material):
    A_val = material['A']
    k_val = material['k']
    T_b = material['T_b']
    alpha = material['alpha']
    C1 = A_val*P / (2 * np.pi * k_val * T_b)
    return C1 * np.log( (a + alpha/v) / a )

def get_max_depth_gs_et(P, v, a, material, T_ambient=0):
    """
    Calculates the melt pool depth using both Gladush-Smurov and Eagar-Tsai,
    returning the maximum of the two to account for conduction-to-keyhole transition.
    """
    
    # 1. Gladush-Smurov Depth (Keyhole/Deep Penetration)
    depth_gs = get_melt_depth_gladush_smurov(P, v, a, material)
    depth_gs = max(0.0, depth_gs) # Prevent negative depths at low energy
    
    # 2. Eagar-Tsai Depth (Conduction)
    # physics.py returns: length, width, depth, x_tail, x_front
    _, _, depth_et, _, _ = get_eagar_tsai_dimensions(
        P, v, a, material, T_ambient=T_ambient, resolution=250
    )
    
    # 3. Take the maximum depth
    hybrid_depth = max(depth_gs, depth_et)
    
    return hybrid_depth, depth_gs, depth_et

## ======================= Defect Criteria ======================= ##

def calculate_melt_pool_dimensions(P, v, material, process_parameters, T_ambient=0, resolution=100):
    """
    Wrapper function that calculates melt pool dimensions using the hybrid approach:
    - Length and Width from Rubenchik's dimensionless model
    - Depth from Gladush-Smurov's deep penetration model
    """
    # Extract laser spot size 'a' from process parameters 
    # (Defaulting to 50 microns if not specified)
    a = process_parameters.get('a', 50e-6)
    
    # 1. Get Length and Width using the Rubenchik dimensionless model
    L, W, _, _, _ = get_rubenchik_dimensions(P, v, a, material, T_ambient=T_ambient, resolution=resolution)
    
    # 2. Get Depth using the maximum of Gladush-Smurov and Eagar-Tsai models
    D, _, _ = get_max_depth_gs_et(P, v, a, material, T_ambient)
    
    # Failsafe: if the Rubenchik model didn't find a melt pool (returns 0 for width/length), 
    # the depth should also be 0 to prevent false defects.
    if W == 0.0 or L == 0.0:
        D = 0.0
        
    return L, W, D

def load_defect_module(module_name):
    """
    Dynamically loads a defect criteria python file from the src/defects folder.
    """
    try:
        # This acts just like 'from src.defects import module_name'
        return importlib.import_module(f"src.defects.{module_name}")
    except ModuleNotFoundError:
        print(f"Warning: Defect module {module_name}.py not found!")
        return None

def compute_printability_map(Power_range, Scan_Speed_range, material, process_parameters, resolution=100, active_defects=None):
    """
    Calculates the grid and evaluates the defects, returning the raw matrices.
    
    active_defects: dict mapping the defect category to the file name, e.g.:
    {'balling': 'ball01', 'lof': 'lof01', 'keyhole': 'key01'}
    """
    if active_defects is None:
        active_defects = {'balling': 'ball01', 'lof': 'lof01', 'keyhole': 'key01'}

    # 1. Dynamically load the requested defect modules
    mod_ball = load_defect_module(active_defects.get('balling'))
    mod_lof = load_defect_module(active_defects.get('lof'))
    mod_key = load_defect_module(active_defects.get('keyhole'))

    # 2. Create the grid
    P_vals = np.linspace(Power_range[0], Power_range[1], resolution)
    v_vals = np.linspace(Scan_Speed_range[0], Scan_Speed_range[1], resolution)
    P_grid, v_grid = np.meshgrid(P_vals, v_vals)
    
    # Initialize the output map
    defect_map = np.zeros((resolution, resolution), dtype=int)

    # 3. Iterate through the grid
    for i in range(resolution):
        for j in range(resolution):
            P_current = P_grid[i, j]
            v_current = v_grid[i, j]

            # Calculate Dimensions
            L, W, D = calculate_melt_pool_dimensions(P_current, v_current, material, process_parameters)
            
            dimensions = {'L': L, 'W': W, 'D': D}
            current_process_params = process_parameters.copy()
            current_process_params['P'] = P_current
            current_process_params['v'] = v_current

            # 4. Evaluate Defect Criteria dynamically
            is_balling = mod_ball.check(dimensions, current_process_params, material) if mod_ball else False
            is_lof = mod_lof.check(dimensions, current_process_params, material) if mod_lof else False
            is_keyhole = mod_key.check(dimensions, current_process_params, material) if mod_key else False

            # 5. Assign to map
            if is_balling:
                defect_map[i, j] = 1
            elif is_lof:
                defect_map[i, j] = 2
            elif is_keyhole:
                defect_map[i, j] = 3
            else:
                defect_map[i, j] = 0 # Safe Zone

    return P_grid, v_grid, defect_map