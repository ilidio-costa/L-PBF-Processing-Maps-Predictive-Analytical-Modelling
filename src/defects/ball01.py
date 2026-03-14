def check(dimensions, process_parameters, material):
    """
    Plateau-Rayleigh Aspect Ratio criterion for Balling.
    Returns True if the track is prone to balling (L/W >= 2.3).
    """
    L = dimensions.get('L', 0)
    W = dimensions.get('W', 0)
    
    if W == 0:
        return True # Avoid division by zero; no width means failure
        
    return (L / W) >= 2.3