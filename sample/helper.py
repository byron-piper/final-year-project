import json
import math

import h5py
import numpy as np

def load_parameters() -> dict:
	with open("parameters.json", "r") as f:
		return json.load(f)
		
def normalise_coords_position(coords:list[np.ndarray], target:np.ndarray) -> np.ndarray:
	combined_coords = np.concat(coords)
	
	max_x = np.max(combined_coords[:, 0])
	mean_y = np.mean(combined_coords[:, 1])
	
	normalised_coords = []
	for coord in coords:
		coord[:, 0] += target[0] - max_x
		coord[:, 1] -= mean_y
		normalised_coords.append(coord)
		
	return normalised_coords
	
def cubic_bezier(t, p0, p1, p2, p3):
	"""
	Returns each point along a given bezier curve controlled by 4-points
	"""
	return (1-t)**3 * p0 + 3*(1-t)**2*t * p1 + 3*(1-t)*t**2 * p2 + t**3 * p3
	
def slope_eq_at_idx(points, idx):
	"""
	Returns both the gradient and y-intercept for a straight line defined between the point at the given index and the previous (or next) point.
	"""
	if idx+1 == len(points):
		m = (points[idx][1] - points[idx-1][1]) / (points[idx][0] - points[idx-1][0])
	else:
		m = (points[idx-1][1] - points[idx][1]) / (points[idx-1][0] - points[idx][0])
	c = points[idx][1] - m*points[idx][0]

	return m, c
	
def closest_point_idx(points, target) -> int:
	"""
	Returns the index of the closest point to some given target within a numpy array
	"""
	if np.isscalar(target):
		return np.argmin(np.abs(points - target))

	distances = np.linalg.norm(points - target, axis=1)
	return np.argmin(distances)
	
def get_bounding_points(coords:np.ndarray) -> tuple[float]:
	btm_left = np.min(coords, axis=0)
	top_right = np.max(coords, axis=0)

	return (btm_left, top_right)
	
def rotate_coords(coords:np.ndarray, aoa:float, pivot:np.ndarray = np.zeros(shape=())) -> np.ndarray:
	aoa_radians = math.radians(aoa)
	rotation_matrix = [
		[math.cos(aoa_radians), math.sin(aoa_radians), 0],
		[-math.sin(aoa_radians), math.cos(aoa_radians), 0],
		[0, 0, 0]
	]

	if pivot.shape == ():
		pivot = np.zeros(shape=coords.shape)

	new_coords = coords - pivot

	new_coords = np.dot(new_coords, rotation_matrix)

	new_coords += pivot
	
	return new_coords

def calc_first_cell_height(Re:float, x:float, rho:float, mu:float, yplus:float):
	U = (Re * mu) / (rho * x)

	cf = 0.026 / (math.pow(Re, 1/7))
	
	tau_w = 0.5 * rho * math.pow(U, 2) * cf
	
	ut = math.pow(tau_w / rho, 0.5)
	
	yh = (yplus * mu) / (ut * rho)
	
	return yh

def calc_freestream_velocity(Re:float, x:float, rho:float, mu:float):
	return (Re * mu) / (x * rho)

def get_mesh_node_count(filename):
    try:
        with h5py.File(filename, "r") as f:
                return len(f["meshes"]["1"]["nodes"]["coords"]["1"])
    except Exception as e:
        print(f"Error reading mesh: '{e}'")