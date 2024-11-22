import math
import json

import gmsh

import parameters
from aerofoil_gen import generate_naca4

class gmsh_wrapper():
    """
    This class acts a proxy to the gmsh library to enable automatic synchronization of the mesh
    """
    def __init__(self, geo):
        self._geo = geo

    def __getattr__(self, name):
        func = getattr(self._geo, name)

        if not callable(func): 
            return func

        def wrapped_func(*args, **kwargs):
            result = func(*args, **kwargs)
            gmsh.model.geo.synchronize()
            return result
        return wrapped_func

def add_aerofoil(geo: gmsh_wrapper, coords: list, aoa: float, mesh_size: float):
    """
    Takes in a list of [x, y, z] coordinates and creates .gmsh describing the aerofoil curve.
    """

    # Rotate coordinates by AoA using rotation matrix:
    #
    # [ cos(AoA)  sin(AoA) ]
    # [ -sin(AoA) cos(AoA) ]
    #
    aoa_radians = math.radians(aoa)
    for i, coord in enumerate(coords):
        x = coord[0] * math.cos(aoa_radians) + coord[1] * math.sin(aoa_radians)
        y = -coord[0] * math.sin(aoa_radians) + coord[1] * math.cos(aoa_radians)
        coords[i] = [x, y, coord[2]]

    points = []
    for coord in coords:
        points.append(geo.addPoint(coord[0], coord[1], coord[2], mesh_size))
    return points, geo.addSpline(points)

def construct_gmsh():
    #region ==== UNPACK PARAMETERS ==== #

    project_id = parameters.project_id

    # Aerofoil parameters
    naca = parameters.naca
    num_points = parameters.num_points
    chord = parameters.chord

    # Gmsh parameters
    mesh_size = parameters.mesh_size
    domain_radius = parameters.domain_radius
    domain_extension = parameters.domain_extension
    AoA = parameters.AoA
    refinement_offset = parameters.refinement_offset

    #endregion

    # ==== INITIALISE GMSH PROJECT ==== #

    gmsh.initialize()
    gmsh.model.add(project_id)

    # Wrap Gmsh to enable automatic synchronization
    geo = gmsh_wrapper(gmsh.model.geo)

    # ==== GENERATE AEROFOIL COORDINATES AND ADD TO MESH ==== #

    aerofoil_coords = generate_naca4(chord=chord, max_camber=naca[0], max_camber_pos=naca[1], thickness=naca[2], num_points=num_points)

    aerofoil_points, _ = add_aerofoil(geo, aerofoil_coords, AoA, mesh_size)

    #region ==== ADD DOMAIN POINTS ==== #

    # Refinement zone
    leading_edge = aerofoil_points[0]
    trailing_edge = aerofoil_points[int(len(aerofoil_points)/2) - 1]
    point_rf_1 = geo.addPoint(0, refinement_offset, 0, mesh_size)
    point_rf_2 = geo.addPoint(chord, refinement_offset, 0, mesh_size)
    point_rf_3 = geo.addPoint(-refinement_offset, 0, 0, mesh_size)
    point_rf_4 = geo.addPoint(0, -refinement_offset, 0, mesh_size)
    point_rf_5 = geo.addPoint(chord, -refinement_offset, 0, mesh_size)

    # Outer domain
    point_dm_1 = geo.addPoint(0, domain_radius, 0, mesh_size)
    point_dm_2 = geo.addPoint(chord, domain_radius, 0, mesh_size)
    point_dm_3 = geo.addPoint(domain_extension, domain_radius, 0, mesh_size)
    point_dm_4 = geo.addPoint(domain_extension, refinement_offset, 0, mesh_size)
    point_dm_5 = geo.addPoint(-domain_radius, 0, 0, mesh_size)
    point_dm_6 = geo.addPoint(domain_extension, 0, 0, mesh_size)
    point_dm_7 = geo.addPoint(domain_extension, -refinement_offset, 0, mesh_size)
    point_dm_8 = geo.addPoint(0, -domain_radius, 0, mesh_size)
    point_dm_9 = geo.addPoint(chord, -domain_radius, 0, mesh_size)
    point_dm_10 = geo.addPoint(domain_extension, -domain_radius, 0, mesh_size)

    #endregion

    #region ==== DRAW LINES ==== #

    # Outer domain (=> refinement zone)
    line_dm_1 = geo.addLine(point_dm_1, point_dm_2)
    line_dm_2 = geo.addLine(point_dm_2, point_dm_3)
    line_dm_3 = geo.addLine(point_dm_1, point_rf_1)
    line_dm_4 = geo.addLine(point_dm_2, point_rf_2)
    line_dm_5 = geo.addLine(point_dm_3, point_dm_4)
    line_dm_6 = geo.addLine(point_rf_1, point_rf_2)
    line_dm_7 = geo.addLine(point_rf_2, point_dm_4)
    #line_dm_8 = geo.addLine(point_rf_2, point_dm_4)
    line_dm_9 = geo.addLine(point_rf_2, trailing_edge)
    line_dm_10 = geo.addLine(point_dm_4, point_dm_6)
    line_dm_11 = geo.addLine(point_dm_5, point_rf_3)
    line_dm_12 = geo.addLine(point_rf_3, leading_edge)
    line_dm_13 = geo.addLine(trailing_edge, point_dm_6)
    #line_dm_14 = geo.addLine(trailing_edge, point_dm_6)
    line_dm_15 = geo.addLine(trailing_edge, point_rf_5)
    line_dm_16 = geo.addLine(point_dm_6, point_dm_7)
    line_dm_17 = geo.addLine(point_rf_4, point_rf_5)
    line_dm_18 = geo.addLine(point_rf_5, point_dm_7)
    line_dm_19 = geo.addLine(point_rf_4, point_dm_8)
    line_dm_20 = geo.addLine(point_rf_5, point_dm_9)
    line_dm_21 = geo.addLine(point_dm_7, point_dm_10)
    line_dm_22 = geo.addLine(point_dm_8, point_dm_9)
    line_dm_23 = geo.addLine(point_dm_9, point_dm_10)

    circ_dm_1 = geo.addCircleArc(point_dm_5, leading_edge, point_dm_1)
    circ_dm_2 = geo.addCircleArc(point_rf_3, leading_edge, point_rf_1)
    circ_dm_3 = geo.addCircleArc(point_rf_4, leading_edge, point_rf_3)
    circ_dm_4 = geo.addCircleArc(point_dm_8, leading_edge, point_dm_5)

    #endregion

    gmsh.write(f"{project_id}.geo_unrolled")
    gmsh.finalize()

def main():
    construct_gmsh()

if __name__ == "__main__":
    main()