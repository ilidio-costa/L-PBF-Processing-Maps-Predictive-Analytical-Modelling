def check(dimensions, process_parameters, material):
    """
    Geometric Ratio criterion for Keyhole porosity.
    Returns True if the melt pool is too deep relative to its width.
    """
    W = dimensions.get('W', 0)
    D = dimensions.get('D', 0)
    
    if D == 0:
        return False # No depth means no keyhole
        
    return (W / D) < 2.0