import numpy as np

def cubic_bezier(t, p0, p1, p2, p3):
    return (1-t)**3 * p0 + 3*(1-t)**2*t * p1 + 3*(1-t)*t**2 * p2 + t**3 * p3

def remove_duplicate_coords(coords: np.ndarray) -> np.ndarray:
    return np.unique(coords, axis=0)