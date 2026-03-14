def check(dimensions, process_parameters, material):
    """
    Geometric Overlap criterion for Lack of Fusion.
    Returns True if voids form between adjacent tracks.
    """
    W = dimensions.get('W', 0)
    D = dimensions.get('D', 0)
    h = process_parameters.get('h', 0)
    t = process_parameters.get('t', 0)
    
    if (t + D) == 0 or W == 0:
        return True 
        
    return ((h / W)**2 + (t / (t + D))) >= 1.0