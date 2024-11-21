import math

import numpy as np

def generate_naca4(chord: float = 1, max_camber: int = 2, max_camber_pos: int = 4, thickness: int = 12, num_points: int = 100, closed_trailing_edge: bool = True, cosine_spacing: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """
    Some documentation.
    """

    M = 9 if max_camber > 9 else max_camber / 100
    P = 9 if max_camber_pos > 9 else max_camber_pos / 10
    T = 40 if thickness > 40 else thickness / 100

    if cosine_spacing:
        beta = np.linspace(0, math.pi, num_points)
        x = (1 - np.cos(beta)) / 2
    else:
        x = np.linspace(0, 1, num_points)

    a4 = -0.1015 if not closed_trailing_edge else -0.1036

    # Generate the thickness distribution
    yt = (T/0.2) * (0.2969*x**0.5 - 0.1260*x - 0.3516*x**2 + 0.2843*x**3 + a4*x**4)
    
    # Generate camber line
    if M > 0:
        yc = np.where(x < P, (M/P**2)*(2*P*x - x**2), (M/(1 - P)**2)*(1 - 2*P + 2*P*x - x**2))
        dyc_dx = np.where(x < P, ((2*M)/P**2)*(P - x), ((2*M) / (1 - P)**2)*(P - x))
    else:
        yc = np.zeros_like(x)
        dyc_dx = np.zeros_like(x)
    
    theta = np.arctan(dyc_dx)

    y = (yc + yt * np.cos(theta)).tolist()
    y_lower = (yc - yt * np.cos(theta)).tolist()

    x = x.tolist()

    x.extend(x[::-1])
    y.extend(y_lower[::-1])

    coords = []
    for i in range(len(x)):
        coords.append([x[i], y[i], 0])

    return coords

def aerofoil_to_3element(chord: float, coords: tuple[np.ndarray, np.ndarray], slat_geom: list, flap_geom: list):
    pass