import numpy as np
from scipy.integrate import fixed_quad
from scipy.optimize import minimize_scalar, brentq

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

## ======================= Gladush-Smurov Model ======================= ##
def get_melt_depth_gladush_smurov(P, v, a, material):
    A_val = material['A']
    k_val = material['k']
    T_b = material['T_b']
    alpha = material['alpha']
    C1 = A_val*P / (2 * np.pi * k_val * T_b)
    return C1 * np.log( (a + alpha/v) / a )

## ======================= Defect Criteria ======================= ##
def get_defect_masks(P, v, a, L, W, D, material, layer_t, hatch_s):
    balling_mask = (np.pi * W / L) <= np.sqrt(2/3)
    
    rho, Cp, Tm, Tb = material['rho'], material['C_p'], material['T_m'], material['T_b']
    alpha, A = material['alpha'], material['A']
    
    norm_enthalpy = (A * P) / (np.pi * rho * Cp * Tm * np.sqrt(alpha * v * a**3))
    keyhole_mask = norm_enthalpy > (np.pi * Tb / Tm)
    
    lof_val = (hatch_s / W)**2 + (layer_t / (layer_t + D))
    lof_mask = lof_val >= 1
    
    return balling_mask, keyhole_mask, lof_mask