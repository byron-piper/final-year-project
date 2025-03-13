import math
import os

import gmsh
import numpy as np
from shapely.geometry import Polygon

from helper import slope_eq_at_idx, closest_coord_idx, cubic_bezier, rotate_coords

def generate_naca4(chord:float=1, M:int=2, P:int=4, T:int=12, num_points:int=100, closed_trailing_edge:bool=True, cosine_spacing:bool=True) -> np.ndarray:
    #region docstring
    """
    Generates and returns (x, y) coordinates for NACA-4 digit aerofoil.
    
    Parameters
    ----------
    chord : float
        Chord length of the aerofoil (default = 1)
    M : int
        Maximum camber of the aerofoil (default = 2)
    P : int
        Position of maximum camber (default = 4)
    T : int
        Maximum thickness of the aerofoil (default = 12)
    num_points : int
        Number of points for upper and lower aerofoil curve (default = 100)
    closed_trailing_edge : bool
        Flag to enable a closed trailing edge (default = True)
    cosine_spacing : bool
        Flag to enable cosine spacing of x-coordinates for group points at curves (default = True)
        
    Returns
    -------
    np.ndarray
        Numpy array of (x, y) coordinates
    """
    #endregion
    
	# Re-scale args to appropriate NACA-4 digits
    M = 0.09 if M > 9 else M / 100
    P = 0.9 if P > 9 else P / 10
    T = 0.4 if T > 40 else T / 100

	# Instantiate 1D vector for x-coordinates, use cosine spacing if requested
    if cosine_spacing:
        beta = np.linspace(0, math.pi, num_points)
        x = (1 - np.cos(beta)) / 2
    else:
        x = np.linspace(0, 1, num_points)

    # Multiply all x-coordinates by chord length
    x *= chord

    # Set 'a4' coefficient depending on closed trailing edge flag
    a4 = -0.1015 if not closed_trailing_edge else -0.1036

    # Generate the thickness distribution
    yt = (T/0.2) * (0.2969*x**0.5 - 0.1260*x - 0.3516*x**2 + 0.2843*x**3 + a4*x**4)

    # Generate camber line of aerofoil
    if M > 0:
        yc = np.where(x < P, (M/P**2)*(2*P*x - x**2), (M/(1 - P)**2)*(1 - 2*P + 2*P*x - x**2))
        dyc_dx = np.where(x < P, ((2*M)/P**2)*(P - x), ((2*M) / (1 - P)**2)*(P - x))
    else:
        yc = np.zeros_like(x)
        dyc_dx = np.zeros_like(x)

    # Calculate theta at each point along camber line
    theta = np.arctan(dyc_dx)

    # [::-1] used to reverse array, [1:-1] used to exclude first and last index else duplicate points will be added
    # Aerofoil is constructed starting from the trailing edge and made anti-clockwise
    x = np.concat((x[::-1], x[1:-1]))
    y = np.concat((yc[::-1] + yt[::-1] * np.cos(theta[::-1]), yc[1:-1] - yt[1:-1] * np.cos(theta[1:-1])))
    coords = np.column_stack((x, y, np.zeros(2*num_points-2)))

    return coords

def generate_slat_coords(chord:float, coords:np.ndarray, slat_geom:dict) -> tuple[np.ndarray, np.ndarray]:
    #region docstring
    """
    Generates (x, y) coordinates for leading edge slat from a given set of (x, y) aerofoil coordinates. Uses `slat_geom`
    dictionary to dictate the respective positioning of bezier curve control and anchor points. Returns a tuple containing
    the generated coordinates and modified aerofoil coordinates with removed leading edge.
    
    Parameters
    ----------
    chord : float
        Chord length of the aerofoil
    coords : np.ndarray
        (x, y) coordinates for aerofoil
    slat_geom : dict
        Dictionary containing parameters dictating positioning and geometric definition of slat
        
    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A tuple containing slat coordinates and modified aerofoil coordinates respectively
    """
    #endregion
    
    # Get the index of the aerofoil's leading edge, located at the minimum x-coordinate
    leading_edge_index = np.argmin(coords[:, 0])

    # Split out x-coordinates into new array
    coords_x = coords[:, 0]

    # Split x-coordinates further into upper and lower surfaces
    coords_x_upr = coords_x[:leading_edge_index+1]
    coords_x_lwr = coords_x[leading_edge_index:]

    # T_x is the upper anchor point on the aerofoil for slat curve
    T_x = slat_geom["Tx"] * chord
    T_idx = closest_coord_idx(coords_x_upr, T_x)
    T_m, T_c = slope_eq_at_idx(coords, T_idx)
    T = np.array([T_x, T_m*T_x + T_c, 0])

    # U_x is the lower anchor point on the aerofoil for slat curve
    U_x = slat_geom["Ux"] * chord
    U_idx = closest_coord_idx(coords_x_lwr, U_x) + leading_edge_index
    U_m, U_c = slope_eq_at_idx(coords, U_idx)
    U = np.array([U_x, U_m*U_x + U_c, 0])

    # V_x is the control point for the slat bezier curve extrending from T_x
    V_x = slat_geom["Vx"] * T_x
    V_m, V_c = slope_eq_at_idx(coords, closest_coord_idx(coords_x_upr, V_x))
    V = np.array([V_x, V_m*V_x + V_c, 0])

    # W_x is the control point for the slat bezier curve extrending from U_x
    W_x = slat_geom["Wx"] * U_x
    W_m, W_c = slope_eq_at_idx(coords, closest_coord_idx(coords_x_lwr, W_x) + leading_edge_index)
    W = np.array([W_x, W_m*W_x + W_c, 0])
    
    # Create new numpy array of points of slat leading edge
    slat_leading_edge = coords[T_idx:U_idx+1]

    # Create Bezier curve subtending points T, V, W, U
    t = np.linspace(0, 1, len(slat_leading_edge))
    bezier_TVWU = np.zeros(shape=(len(slat_leading_edge), 3))
    for i, ti in enumerate(t):
        bezier_TVWU[i] = cubic_bezier(ti, T, V, W, U)

    # Apply cross product on surface tangents at slat leading edge endpoint and aft curve start point.
    # If the result is less than one, then `slat_coords` appends the aft curve without the final coordinate.
    if np.cross(slat_leading_edge[-1] - slat_leading_edge[-2], bezier_TVWU[-1] - slat_leading_edge[-1])[2] < 0:
        temp = bezier_TVWU[:-1]
        slat_coords = np.concat((slat_leading_edge, temp[::-1]))
    else:
        slat_coords = np.concat((slat_leading_edge, bezier_TVWU[::-1]))

    # Create a numpy array storing the modified aerofoil coordinates without the slat geometry
    coords_wo_slat = np.concat((coords[:T_idx+1], bezier_TVWU, coords[U_idx:]))

    # Remove points at sharp edges within small radius to prevent extremely small edges
    slat_coords = unsharpen_coord_inflection(slat_coords, 0, 0.0075)
    slat_coords = unsharpen_coord_inflection(slat_coords, len(slat_leading_edge)-3, 0.01)

    #region # ==== APPLY TRANSFORMATIONS ==== #

    # Apply deflection rotation
    slat_coords = rotate_coords(slat_coords, slat_geom["deflection"], T)

    # Shift slat coordinates to given slat x offset
    aerofoil_le_x = np.min(bezier_TVWU, axis=0)[0]
    delta_x = -np.max(slat_coords, axis=0)[0] + aerofoil_le_x + slat_geom["x_offset"] * chord
    slat_coords += np.array((delta_x, 0, 0))

    # T_prime is defined as the position T now exists at given the previous two transformations
    T_prime = slat_coords[-1]
    T_m_prime, T_c_prime = slope_eq_at_idx(bezier_TVWU, closest_coord_idx(bezier_TVWU[bezier_TVWU[:, 1] > 0], T_prime))
    T_prime_theta = math.atan(1 / T_m_prime)
    delta_y = (slat_geom["y_offset"] * chord - (T_prime[1] - (T_m_prime*T_prime[0] + T_c_prime))) * math.sin(T_prime_theta)
    slat_coords += np.array((0, delta_y, 0))
    
    slat_coords = np.concat((slat_coords, [slat_coords[0]]))
    
    #endregion

    return slat_coords, coords_wo_slat

def generate_flap_coords(chord:float, coords:np.ndarray, flap_geom:dict) -> tuple[np.ndarray, dict]:
	#region docstring
    """
    Generates (x, y) coordinates for trailing edge fowler flap from a given set of (x, y) aerofoil coordinates. Uses `flap_geom`
    dictionary to dictate the respective positioning of bezier curve control and anchor points. Returns a tuple containing
    the generated coordinates and modified aerofoil coordinates with removed trailing edge.
    
    Parameters
    ----------
    chord : float
        Chord length of the aerofoil
    coords : np.ndarray
        (x, y) coordinates for aerofoil
    flap_geom : dict
        Dictionary containing parameters dictating positioning and geometric definition of flap
        
    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A tuple containing flap coordinates and modified aerofoil coordinates respectively
    """
    #endregion
 
    # Get the index of the aerofoil's leading edge, located at the minimum x-coordinate
    leading_edge_idx = np.argmin(coords[:, 0])

    # Split out x-coordinates into new arrayw
    coords_x = coords[:, 0]

    # Split x-coordinates further into upper and lower surfaces
    coords_x_upr = coords_x[:leading_edge_idx+1]
    coords_x_lwr = coords_x[leading_edge_idx:]

    # Determine Point-A by extrapolating from nearest point on aerofoil
    B_x = flap_geom["Bx"] * chord
    A_m, A_c = slope_eq_at_idx(coords, closest_coord_idx(coords_x_upr, B_x))
    A = np.array([B_x, A_m*B_x + A_c, 0])

    # Determine Point-C
    C_x = B_x + flap_geom["Cx"] * (1 - B_x)
    C_idx = closest_coord_idx(coords_x_upr, C_x)
    C_m, C_c = slope_eq_at_idx(coords, C_idx)
    C = np.array([C_x, C_m*C_x + C_c, 0])

    # Determine Point-D by extrapolating from nearest point on aerofoil
    D_m, D_c = slope_eq_at_idx(coords, closest_coord_idx(coords_x_lwr, B_x) + leading_edge_idx)
    D = np.array([B_x, D_m*B_x + D_c, 0])

    # Determine Point-B
    B = np.array([B_x, D[1] + flap_geom["By"] * (A[1] - D[1]), 0])

    # Determine Point-E
    E = coords[-1] # Can be ignored as is not needed

    # Determine Point-F
    F_x = B_x + flap_geom["Fx"] * (1 - B_x)
    F_idx = closest_coord_idx(coords_x_lwr, F_x) + leading_edge_idx
    F_m, F_c = slope_eq_at_idx(coords, F_idx)
    F = np.array([F_x, F_m*F_x + F_c, 0])

    # Determine Point-G
    G_x = D[0] + flap_geom["Gx"] * (F[0] - D[0])
    G_m, G_c = slope_eq_at_idx(coords, closest_coord_idx(coords_x_lwr, G_x) + leading_edge_idx)
    G = np.array([G_x, G_m*G_x + G_c, 0])

    # Determine Point-L
    L = np.array([B[0], B[1] + flap_geom["Ly"] * (D[1] - B[1]), 0])

    # Determine Point-M
    M = np.array([B[0], B[1] + flap_geom["My"] * (A[1] - B[1]), 0])

    # Determine Point-N
    N_x = A[0] + flap_geom["Nx"] * (C[0] - A[0])
    N_m, N_c = slope_eq_at_idx(coords, closest_coord_idx(coords_x_upr, N_x))
    N = np.array([N_x, N_m*N_x + N_c, 0])

    # Create array of t values used to create bezier curve
    t = np.linspace(0, 1, 25)

    # Create Bezier curve subtending points B, M, N, C and points B, L, G, F
    bezier_BMNC = np.zeros(shape=(len(t), 3))
    bezier_BLGF = np.zeros(shape=(len(t), 3))
    for i, ti in enumerate(t):
        bezier_BMNC[i] = cubic_bezier(ti, B, M, N, C)
        bezier_BLGF[i] = cubic_bezier(ti, B, L, G, F)

    # Determine Point-P
    P_x = B[0] + flap_geom["Px"] * (C[0] - B[0])
    P_idx = closest_coord_idx(bezier_BMNC[:, 0], P_x)
    P_m, P_c = slope_eq_at_idx(bezier_BMNC, P_idx)
    P = np.array([P_x, P_m*P_x + P_c, 0])

    # Determine Point-S
    S_x = flap_geom["Sx"] * chord
    S_idx = closest_coord_idx(coords_x_lwr, S_x) + len(coords_x_upr)
    S_m, S_c = slope_eq_at_idx(coords, S_idx)
    S = np.array([S_x, S_m*S_x + S_c, 0])

    # Determine Point-P1
    P1_x = S[0] + flap_geom["P1x"] * (P[0] - S[0])
    P_m, P_c = slope_eq_at_idx(bezier_BMNC, closest_coord_idx(bezier_BMNC[:, 0], P[0]))
    P1 = np.array([P1_x, P_m*P1_x + P_c, 0])

    # Determine Point-S1
    S1_x = S[0] + flap_geom["S1x"] * (P[0] - S[0])
    S_m, S_c = slope_eq_at_idx(coords, closest_coord_idx(coords_x_lwr, S[0]) + leading_edge_idx)
    S1 = np.array([S1_x, S_m*S1_x + S_c, 0])

    # Create Bezier curve subtending points S, S1, P1, P
    bezier_SS1P1P = np.zeros(shape=(len(t), 3))
    for i, ti in enumerate(t):
        bezier_SS1P1P[i] = cubic_bezier(ti, S, S1, P1, P)

    # Point F is created by determining a new point closest to some existing point by extrapolating along a straight line defined by nearest points on aerofoil
    # Therefore, F_idx does not guarantee a realistic index and may point to a point that should not exist on the curve. This simple if statement
    # accounts for this
    if F[0] > coords[F_idx][0]:
        F_idx += 1

    # Create array storing generated flap coordinates
    flap_coords = np.concat((coords[:C_idx], bezier_BMNC[::-1][:-1], bezier_BLGF, coords[F_idx:]))

    # Create a numpy array storing the modified aerofoil coordinates without the flap geometry
    coords_wo_flap = np.concat((coords[C_idx:S_idx], bezier_SS1P1P, bezier_BMNC[P_idx:][1:]))

    # Remove points at sharp edges within small radius to prevent extremely small edges
    coords_wo_flap = unsharpen_coord_inflection(coords_wo_flap, 0, 0.05)
    flap_coords = unsharpen_coord_inflection(flap_coords, 0, 0.005)

    #region # ==== APPLY TRANSFORMATIONS ==== #

    flap_coords = rotate_coords(flap_coords, -flap_geom["deflection"], B)

    # Update C_x as coordinates were removed
    C_x = np.max(coords_wo_flap[:, 0])

    delta_x = -B_x + C_x - flap_geom["x_offset"]
    flap_coords += np.array((delta_x, 0, 0))

    # Update C as coordinates were removed
    C = coords_wo_flap[np.argmax(coords_wo_flap[:, 0])]

    C_m_prime, C_c_prime = slope_eq_at_idx(flap_coords, closest_coord_idx(flap_coords, C))
    C_prime_theta = math.atan(-1 / C_m_prime)
    delta_y = (flap_geom["y_offset"] * chord - (C[1] - (C_m_prime*C[0] + C_c_prime))) * math.sin(C_prime_theta)
    flap_coords += np.array((0, delta_y, 0))

    #endregion

    coords_wo_flap = np.concat((coords_wo_flap, [coords_wo_flap[0]]))
    flap_coords = np.concat((flap_coords, [flap_coords[0]]))

    return flap_coords, coords_wo_flap

def normalise_aerofoil_coords(coords:list[np.ndarray], target_x:float) -> list[np.ndarray]:
    #region docstring
    """
    Normalises a given list aerofoil coordinates by applying the following operations:
        - Shifts the coordinates along the x-axis towards the given `target_x` value
        - Shifts the coordinates along the y-axis towards the overall vertical mean of
          all coordinates
    Returns a new list containing translated (normalised) aerofoil coordinates.
    
    Parameters
    ----------
    coords : list[np.ndarray]
        List of numpy arrays containing (x, y) coordinates of aerofoils/aerofoil-elements
    target_x : float
        Target x-coordinate to shift all coordinates towards
        
    Returns
    -------
    list[np.ndarray]
        A list of translated (normalised) aerofoil coordinates
    """
    #endregion
    
    # Combine all coordinates into a single numpy array
    combined_coords = np.concat(coords)

    # Finds the maximum x-coordinate and average y-coordinate from combined coords
    max_x = np.max(combined_coords[:, 0])
    mean_y = np.mean(combined_coords[:, 1])

    # Loop through each coord and apply translations
    normalised_coords = []
    for coord in coords:
        coord[:, 0] += target_x - max_x
        coord[:, 1] -= mean_y
        normalised_coords.append(coord)
        
    return normalised_coords

def save_highlift_aerofoil(aerofoil_id:str, aerofoil_coords:list[np.ndarray], directory:str) -> None:
    #region docstring
    """
    Saves a high-lift aerofoil with a given `aerofoil_id` and list of `aerofoil_coords` to specified root `directory`
    into a folder with given `aerofoil_id`. Coordinates are saved as `.txt` files in the SpaceClaim polyline
    point format.
    
    Parameters
    ----------
    aerofoil_id : str
        Identifier for aerofoil
    aerofoil_coords : list[np.ndarray]
        List of numpy (x, y) coordinates for aerofoil elements (base, flap, slat)
    directory : str
        Absolute path to save directory
        
    Returns
    -------
    None
    """
    #endregion
    
    # Convert each entry of aerofoil coordinates to given SpaceClaim polyine point format
    # Example: `1   0.324353    0.785245`
    aerofoil_coords[0] = [f"1\t{row[0]}\t{row[1]}\n" for row in aerofoil_coords[0]]
    aerofoil_coords[1] = [f"1\t{row[0]}\t{row[1]}\n" for row in aerofoil_coords[1]]
    aerofoil_coords[2] = [f"1\t{row[0]}\t{row[1]}\n" for row in aerofoil_coords[2]]

    # Intialise folder path and create directory if it doesn't exist already
    aerofoil_folder = os.path.join(directory, aerofoil_id)
    if not os.path.exists(aerofoil_folder):
        os.mkdir(aerofoil_folder)

    # Loop through each aerofoil element and save coordinates to respective `.txt` file
    for i, element in enumerate(["base", "flap", "slat"]):
        filepath = os.path.join(aerofoil_folder, f"{element}.txt")
        with open(filepath, "w") as f:
            f.write("polyline=true\n")
            f.writelines(aerofoil_coords[i])

def unsharpen_coord_inflection(coords:np.ndarray, idx:int, radius:float) -> np.ndarray:
    #region docstring
    """
    Unsharpens coordinate inflection point by removing all coordinates within a specified radius
    at a given point.
    
    Parameters
    ----------
    coords : np.ndarray
        Numpy array containing (x, y) coordinates
    idx : int
        Index of inflection point
    radius : float
        Radius around inflection point to remove coordinates
        
    Returns
    -------
    np.ndarray
        Numpy array containing (x, y) coordinates with removed points
    """
    #endregion
    
    # Sqaure distances of all coordinates relative to inflection point
    distances_sq = np.sum((coords - coords[idx]) ** 2, axis=1)

    # Indicies mask for all distances greater than a square of the given radius
    mask = distances_sq > radius**2

    return coords[mask]

def aerofoil_elements_intersecting(base_coords:np.ndarray, flap_coords:np.ndarray, slat_coords:np.ndarray) -> bool:
    #region docstring
    """
    Returns a `bool` value depending on whether each aerofoil element coordinates are intersecting
    
    Parameters
    ----------
    base_coords : np.ndarray
        Numpy array containing (x, y) coordinates for base element
    flap_coords : np.ndarray
        Numpy array containing (x, y) coordinates for flap element
    slat_coords : np.ndarray
        Numpy array containing (x, y) coordinates for slat element
        
    Returns
    -------
    bool
        Flag stating if any aerofoil element coordinates are intersecting
    """
    #endregion
    
    # Create polygons from given coordinates for each element
    base_polygon = Polygon(base_coords)
    flap_polygon = Polygon(flap_coords)
    slat_polygon = Polygon(slat_coords)

    # Perform intersection checks between each polygon, `True` if two set of coordinates
    # are intersecting, else `False`
    base_intersects_flap = base_polygon.intersects(flap_polygon)
    base_intersects_slat = base_polygon.intersects(slat_polygon)
    slat_intersects_flap = slat_polygon.intersects(flap_polygon)

    return base_intersects_flap or base_intersects_slat or slat_intersects_flap

def visualise_aerofoil_coords(aerofoils:dict, tag:str=None, columns:int=3, open_gmsh:bool=True) -> None:
    #region docstring
    """
    Creates a GMSH `.geo_unrolled` file to visualise a dictionary of aerofoils. Coordinates for each aerofoil
    are displayed in uniform grid
    
    Parameters
    ----------
    aerofoils : dict
        Dictionary containing aerofoils with given ID and element (x, y) coordinates
    tag : str, optional
        Tag to specify an exact aerofoil to visualise, else all aerofoils will be visualised
    columns : int
        Number of columns in uniform (default = 3)
    open_gmsh : bool
        Flag to enable opening GMSH window (default = True)
        
    Returns
    -------
    None
    """
    #endregion
    
    gmsh.initialize("aerofoils_visualised")

    # Visualise specific aerofoil
    if tag:
        for element in aerofoils[tag]:
            for coord in element:
                gmsh.model.geo.addPoint(coord[0], coord[1], 0, 1)
        gmsh.model.geo.synchronize()
        gmsh.write("aerofoils_visualised.geo_unrolled")
        if open_gmsh: gmsh.fltk.run()
        gmsh.finalize()
        return

    # Visualise all aerofoils
    row = 0
    for i, (_, elements) in enumerate(aerofoils.items()):
        if i > 0 and i % columns == 0: row += 1

        for element in elements:
            for coord in element:
                gmsh.model.geo.addPoint(coord[0] + (2*(i-row*columns)), coord[1] - (2*row), 0, 1)

        gmsh.model.geo.synchronize()

    # Create `visualisation` folder if it doesn't exist already
    if not os.path.exists("visualisation"):
        os.mkdir("visualisation")

    gmsh.write("visualisation/aerofoils_visualised.geo_unrolled")
    if open_gmsh: gmsh.fltk.run()
    gmsh.finalize()