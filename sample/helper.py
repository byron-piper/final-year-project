import json
import math

import h5py
import numpy as np

def load_parameters() -> dict:
    #region docstring
    """
    Returns a dictionary read from `parameters.json`
    
    Parameters
    ----------
    None
        
    Returns
    -------
    dict
        Dictionary containing parameters
    """
    #endregion
    
    with open("parameters.json", "r") as f:
        return json.load(f)
	
def cubic_bezier(t, p0, p1, p2, p3) -> np.ndarray:
    #region docstring
	"""
	Returns coordinates for a generated cubic bezier curve
    
    Parameters
    ----------
    t : float
        Position between control points `p0` and `p1` with range {0, 1}
    p0 : np.ndarray
        Vector containing (x, y) coordinates of first anchor point
    p1 : np.ndarray
        Vector containing (x, y) coordinates of first control point
    p2 : np.ndarray
        Vector containing (x, y) coordinates of second control point
    p3 : np.ndarray
        Vector containing (x, y) coordinates of second anchor point
        
    Returns
    -------
    np.ndarray
        Numpy array containing (x, y) coordinates of cubic bezier curve
	"""
    #endregion
 
	return (1-t)**3 * p0 + 3*(1-t)**2*t * p1 + 3*(1-t)*t**2 * p2 + t**3 * p3
	
def slope_eq_at_idx(coords:np.ndarray, idx:int) -> np.ndarray:
    #region docstring
    """
	Returns both the gradient and y-intercept of a straight line approximated between `points`
    near given `idx`
    
    Parameters
    ----------
    coords : np.ndarray
        Numpy array containing (x, y) coordinates
    idx : int
        Index in coordinate array to approximate straight line
        
    Returns
    -------
    tuple[float, float]
        Tuple containing the gradient and y-intercept respectively
	"""
    #endregion
    
    # If `idx` at the end of the coordinate array, loop back to the start
    if idx+1 == len(coords):
        m = (coords[idx][1] - coords[idx-1][1]) / (coords[idx][0] - coords[idx-1][0])
    else:
        m = (coords[idx-1][1] - coords[idx][1]) / (coords[idx-1][0] - coords[idx][0])
    c = coords[idx][1] - m*coords[idx][0]

    return m, c
	
def closest_coord_idx(coords:np.ndarray, target:np.ndarray) -> int:
    #region docstring
	"""
	Returns the index of the closest coordinate to a given target vector
    
    Parameters
    ----------
    coords : np.ndarray
        Numpy array containing (x, y) coordinates
    target : np.ndarray
        Numpy array containing (x, y) target point
        
    Returns
    -------
    int
        Index of the closest point to coordinates array
	"""
    #endregion
 
    # Return index if target is a scalar value (non-vector)
	if np.isscalar(target):
		return np.argmin(np.abs(coords - target))

	distances = np.linalg.norm(coords - target, axis=1)
	return np.argmin(distances)
	
def get_bounding_points(coords:np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    #region docstring
    """
    Returns a tuple containing two vectors containing the positions of the
    bottom left and top right bounding points enveloping the coordinates array
    
    Parameters
    ----------
    coords : np.ndarray
        Numpy array containing (x, y) coordinates
        
    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Tuple containing two vectors of bounding points
    """
    #endregion
    
    btm_left = np.min(coords, axis=0)
    top_right = np.max(coords, axis=0)

    return (btm_left, top_right)
	
def rotate_coords(coords:np.ndarray, angle:float, pivot:np.ndarray = np.zeros(shape=())) -> np.ndarray:
    #region docstring
    """
    Rotates a given Numpy array of (x, y) coordinates by `angle` relative to some `pivot` vector
    
    Parameters
    ----------
    coords : np.ndarray
        Numpy array containing (x, y) coordinates
    angle : float
        Rotation angle in degrees
    pivot : np.ndarray
        Pivot vector to rotate coordinates relative to (default = [0, 0, 0])
        
    Returns
    -------
    np.ndarray
        Numpy array containing rotated coordinates
    """
    #endregion
    
    angle_radians = math.radians(angle)
    rotation_matrix = [
        [math.cos(angle_radians), math.sin(angle_radians), 0],
        [-math.sin(angle_radians), math.cos(angle_radians), 0],
        [0, 0, 0]
    ]

    if pivot.shape == ():
        pivot = np.zeros(shape=coords.shape)

    new_coords = coords - pivot

    new_coords = np.dot(new_coords, rotation_matrix)

    new_coords += pivot
	
    return new_coords

def translate_coords(coords:np.ndarray, target:np.ndarray, origin:np.ndarray = None) -> np.ndarray:
    #region docstring
    """
    Shifts a set of (x, y) coordinates towards a given target point relative to some origin point.
    If no origin is provided, coordinates are shifted relative to the centroid of the coordinates.

    Parameters
    ----------
    coords : np.ndarray
        A 2D numpy array of shape (N, 2) representing (x, y) coordinates.
    target : np.ndarray
        A 1D numpy array of shape (1, 2) representing the target (x, y) coordinates.
    origin : np.ndarray, optional
        A 1D numpy array of shape (1, 2) representing the origin point. 
        If None, the centroid of `coords` is used as the origin.

    Returns
    -------
    np.ndarray
        A 2D numpy array of the same shape as `coords`, with coordinates shifted toward the target.
    """
    #endregion
    
    if not origin: origin = np.mean(coords, axis=0)
    
    translation_vector = target - origin
    return coords + translation_vector

def calc_first_cell_height(Re:float, x:float, rho:float, mu:float, yplus:float) -> float:
    #region docstring
    """
    Calculates the first mesh cell height of the boundary layer
    
    Parameters
    ----------
    Re : float
        Reynolds number of flow
    x : float
        Chord / length
    rho : float
        Freestream density
    mu : float
        Freestream viscosity
    yplus : float
        Target CFD y-plus value
        
    Returns
    -------
    float
        First cell height of mesh boundary layer
    """
    #endregion
    
    U = (Re * mu) / (rho * x)

    cf = 0.026 / (math.pow(Re, 1/7))

    tau_w = 0.5 * rho * math.pow(U, 2) * cf

    ut = math.pow(tau_w / rho, 0.5)

    yh = (yplus * mu) / (ut * rho)

    return yh

def calc_freestream_velocity(Re:float, x:float, rho:float, mu:float) -> float:
    #region docstring
    """
    Calculates the freestream velocity from at Reynolds number `Re`
    
    Uses equation:
        :math:`\frac{Re\times\mu}{x\times\rho}`
    
    Parameters
    ----------
    Re : float
        Reynolds number of flow
    x : float
        Chord / length
    rho : float
        Freestream density
    mu : float
        Freestream viscosity
        
    Returns
    -------
    float
        First cell height of mesh boundary layer
    """
    #endregion
    
    return (Re * mu) / (x * rho)

def get_mesh_node_count(filename:str) -> None:
    #region docstring
    """
    Gets the total node count from a given `.h5` mesh file
    
    Parameters
    ----------
    filename : str
        Filename of `.h5` mesh file
        
    Returns
    -------
    None
    
    Raises
    -------
    Exception
        Failure to read mesh and extract node count
    """
    #endregion
    
    try:
        with h5py.File(filename, "r") as f:
                return len(f["meshes"]["1"]["nodes"]["coords"]["1"])
    except Exception as e:
        print(f"Error reading mesh: '{e}'")