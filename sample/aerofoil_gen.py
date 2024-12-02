import math
import logging

import numpy as np

from helper import cubic_bezier

def generate_naca4(chord: float = 1, M: int = 2, P: int = 4, T: int = 12, num_points: int = 100, closed_trailing_edge: bool = True, cosine_spacing: bool = True) -> np.ndarray:
    """
    Some documentation.
    """
    
    logging.info(f"Generating NACA-4 aerofoil with configuration:\n \
                 chord = {chord}, M = {M}, P = {P}, T = {T}, num_points = {num_points}")

    # Re-scale NACA-4 digits to appropriate digits
    M = 0.09 if M > 9 else M / 100
    P = 0.9 if P > 9 else P / 10
    T = 0.4 if T > 40 else T / 100

    # Instantiate 1D vector for x, use cosine spacing if requested
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

    # [::-1] used to reverse array, [1:-1] used to exclude first and last index else duplicate points will be added
    # Aerofoil is constructed starting from the trailing edge and made anti-clockwise
    x = np.concat((x[::-1], x[1:-1]))
    y = np.concat((yc[::-1] + yt[::-1] * np.cos(theta[::-1]), yc[1:-1] - yt[1:-1] * np.cos(theta[1:-1])))

    coords = np.column_stack((x, y, np.zeros(2*num_points-2)))

    return coords

def aerofoil_to_3element(chord: float, coords: np.ndarray, slat_geom: list, flap_geom: list) -> np.ndarray:
    logging.info(f"Converting aerofoil to 3-element:\n \
                 Slat geometry = {slat_geom}\n \
                 Flap geometry = {flap_geom}")
    
    coords_half_length = int(len(coords)/2)

    # Unpack coordinates into x, y vectors
    x = coords[:, 0]
    y = coords[:, 1]
    
    #region # ==== CREATE SLAT COORDINATES ==== #

    # Get x-limit for each slat endpoints
    slat_upr_xlim = chord * slat_geom[0]
    slat_lwr_xlim = chord * slat_geom[0] * slat_geom[1]

    # Get indicies of slat endpoints
    slat_upr_xidx = np.argmin(np.abs(x[y >= 0] - slat_upr_xlim))
    slat_lwr_xidx = np.argmin(np.abs(x[y < 0] - slat_lwr_xlim)) + coords_half_length

    # Get coordinates for P3 & P0, start and end points of bezier curve (representing the slat inner curve)
    slat_P3 = coords[slat_upr_xidx]
    slat_P0 = coords[slat_lwr_xidx]

    # Get slope at P3 & P0, used to extrapolate control points along tangent at the aerofoil's surface from given points
    slat_upr_slope = (coords[slat_upr_xidx+1][1] - coords[slat_upr_xidx-1][1]) / \
                     (coords[slat_upr_xidx+1][0] - coords[slat_upr_xidx-1][0])
    slat_lwr_slope = (coords[slat_lwr_xidx+1][1] - coords[slat_lwr_xidx-1][1]) / \
                     (coords[slat_lwr_xidx+1][0] - coords[slat_lwr_xidx-1][0])

    # Get constant 'c' of tangent line equations for P3 & P0
    slat_upr_c = slat_P3[1] - slat_upr_slope*slat_P3[0]
    slat_lwr_c = slat_P0[1] - slat_lwr_slope*slat_P0[0]

    # Get x-position for P2 & P1, control points for the bezier curve (representing the slat inner curve)
    slat_P2_x = slat_upr_xlim - (slat_geom[2] * slat_upr_xlim)
    slat_P1_x = slat_lwr_xlim - (slat_geom[3] * slat_lwr_xlim)

    # Generate coordinates for both P2 & P1 using the slope and constant 'c' of the control lines
    slat_P2 = np.array([slat_P2_x, slat_upr_slope*(slat_P2_x) + slat_upr_c, 0])
    slat_P1 = np.array([slat_P1_x, slat_lwr_slope*(slat_P1_x) + slat_lwr_c, 0])

    # Slice given coordinates to get the slat leading edge (outer frontal curve) given the two endpoints
    slat_leading_edge = coords[slat_upr_xidx:slat_lwr_xidx+1]

    # Initialise new coordinate matrix for the trailing edge points, with the same shape as the leading edge
    num_slat_points = len(slat_leading_edge)
    slat_trailing_edge = np.zeros(shape=slat_leading_edge.shape)

    # Generate coordinates throughout the bezier curve
    slat_inner_t = np.linspace(0, 1, num_slat_points)
    for i, t in enumerate(slat_inner_t):
        slat_trailing_edge[i] = cubic_bezier(t, slat_P0, slat_P1, slat_P2, slat_P3)

    # Concatenate the leading and trailing edge coordinates into a single 2D matrix
    slat_coords = np.concat((slat_leading_edge, slat_trailing_edge[1:-1]))

    # Update the aerofoil coords to remove the slat coordinates
    new_aerofoil_coords = np.concat((coords[:slat_upr_xidx+1], slat_trailing_edge[::-1][1:-1], coords[slat_lwr_xidx:]))

    #endregion

    #region # ==== CREATE FLAP COORDINATES ==== #

    # Get x-limit for each slat endpoints
    flap_upr_xlim = 1 - (chord * flap_geom[0] * flap_geom[1])
    flap_lwr_xlim = 1 - (chord * flap_geom[0])

    # Get indicies of slat endpoints
    flap_upr_xidx = np.argmin(np.abs(x[y >= 0] - flap_upr_xlim))
    flap_lwr_xidx = np.argmin(np.abs(x[y < 0] - flap_lwr_xlim)) + coords_half_length

    # Get coordinates for P3 & P0, start and end points of bezier curve (representing the flap outer curve)
    P3 = coords[flap_upr_xidx]
    P0 = coords[flap_lwr_xidx]

    # Get slope at P3 & P0, used to extrapolate control points along tangent at the aerofoil's surface from given points
    flap_upr_slope = (coords[flap_upr_xidx+1][1] - coords[flap_upr_xidx-1][1]) / \
                     (coords[flap_upr_xidx+1][0] - coords[flap_upr_xidx-1][0])
    flap_lwr_slope = (coords[flap_lwr_xidx+1][1] - coords[flap_lwr_xidx-1][1]) / \
                     (coords[flap_lwr_xidx+1][0] - coords[flap_lwr_xidx-1][0])

    # Get constant 'c' of tangent line equations for P3 & P0
    flap_upr_c = P3[1] - flap_upr_slope*P3[0]
    flap_lwr_c = P0[1] - flap_lwr_slope*P0[0]

    # Get x-position for P2 & P1, control points for the bezier curve (representing the flap outer curve)
    flap_xP2_upr = flap_upr_xlim - (flap_geom[2] * flap_upr_xlim)
    flap_xP1_upr = flap_lwr_xlim - (flap_geom[3] * flap_lwr_xlim)

    # Generate coordinates for both P2 & P1 using the slope and constant 'c' of the control lines
    P2 = np.array([flap_xP2_upr, flap_upr_slope*(flap_xP2_upr) + flap_upr_c, 0])
    P1 = np.array([flap_xP1_upr, flap_lwr_slope*(flap_xP1_upr) + flap_lwr_c, 0])

    # Slice given coordinates to get the flap leading edge (outer frontal curve) given the two endpoints
    flap_trailing_coords = np.concat((new_aerofoil_coords[flap_lwr_xidx:], new_aerofoil_coords[:flap_upr_xidx+1]))

    # Initialise new coordinate matrix for the trailing edge points, with the same shape as the leading edge
    num_flap_points = len(flap_trailing_coords)
    flap_leading_coords = np.zeros(shape=flap_trailing_coords.shape)

    # Generate coordinates throughout the bezier curve
    flap_t = np.linspace(0, 1, num_flap_points)
    for i, t in enumerate(flap_t):
        flap_leading_coords[i] = cubic_bezier(t, P0, P1, P2, P3)

    # Concatenate the leading and trailing edge coordinates into a single 2D matrix
    flap_coords = np.concat((flap_trailing_coords, flap_leading_coords[1:-1][::-1]))

    # Update the aerofoil coords to remove the slat coordinates
    new_aerofoil_coords = np.concat((new_aerofoil_coords[flap_upr_xidx:flap_lwr_xidx], flap_leading_coords))

    #endregion

    return new_aerofoil_coords, slat_coords, flap_coords, P0, P1, P2, P3