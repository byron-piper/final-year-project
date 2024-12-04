import math
import logging

import numpy as np
import gmsh

from helper import remove_duplicate_coords
import parameters as params
from aerofoil_gen import generate_naca4, aerofoil_to_3element, rotate_element

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

def add_aerofoil(geo: gmsh_wrapper, coords: list, mesh_size: float, total_lines:int, loop: bool = True):
    """
    Takes in a list of [x, y, z] coordinates and creates .gmsh describing the aerofoil curve.
    """

    # Rotate coordinates by AoA using rotation matrix:
    #
    # [ cos(AoA)  sin(AoA) ]
    # [ -sin(AoA) cos(AoA) ]
    #

    start_line_idx = total_lines + 1

    points = []
    lines = []
    for i, coord in enumerate(coords):
        points.append(geo.addPoint(coord[0], coord[1], coord[2], mesh_size))
        if i > 0: 
            lines.append(geo.addLine(points[i - 1], points[i]))
    if loop: 
        lines.append(geo.addLine(points[0], points[-1]))

    total_lines = len(lines)

    return points, total_lines, (start_line_idx, total_lines)

def construct_gmsh():
    total_lines = 0
    #region ==== UNPACK PARAMETERS ==== #

    project_id = params.project_id

    # Aerofoil parameters
    naca = params.naca
    num_points = params.num_points
    chord = params.chord

    # High Lift Devices parameters
    slat_geom = params.slat_geom
    flap_geom = params.flap_geom

    # Gmsh parameters
    mesh_size = params.mesh_size
    domain_radius = params.domain_radius
    domain_extension = params.domain_extension
    AoA = params.AoA
    refinement_offset = params.refinement_offset

    #endregion

    # ==== INITIALISE GMSH PROJECT ==== #

    logging.info(f"Initialising GMSH with Project ID = {project_id}")

    gmsh.initialize()
    gmsh.model.add(project_id)

    # Wrap Gmsh to enable automatic synchronization
    geo = gmsh_wrapper(gmsh.model.geo)

    # ==== GENERATE AEROFOIL COORDINATES AND ADD TO MESH ==== #

    aerofoil_coords = generate_naca4(chord=chord, M=naca[0], P=naca[1], T=naca[2], num_points=num_points)

    aerofoil_coords = rotate_element(aerofoil_coords, AoA)
    
    aerofoil_coords, slat_coords, flap_coords = aerofoil_to_3element(chord, aerofoil_coords, slat_geom, flap_geom)

    logging.info(f"Transforming aerofoil, slat and flap coordinates")

    #slat_coords = translate_aerofoil(slat_coords, offsets["slat"])
    #flap_coords = translate_aerofoil(flap_coords, offsets["flap"])

    #slat_coords = translate_aerofoil(slat_coords, offsets["slat"])

    #aerofoil_coords = rotate_aerofoil(aerofoil_coords, AoA)
    #slat_coords = rotate_aerofoil(slat_coords, AoA)
    #flap_coords = rotate_aerofoil(flap_coords, AoA)

    logging.info(f"Adding aerofoil elements to GMSH .geo file")

    aerofoil_points, total_lines, (aerofoil_start, aerofoil_end) = add_aerofoil(geo, aerofoil_coords, mesh_size, total_lines)

    slat_points, total_lines, (slat_start, slat_end) = add_aerofoil(geo, slat_coords, mesh_size, total_lines)
    flap_points, total_lines, (flap_start, flap_end) = add_aerofoil(geo, flap_coords, mesh_size, total_lines)

    if params.debug_enabled:
        logging.debug(f"Drawing bezier control lines")

        P0_point = geo.addPoint(P0[0], P0[1], 0)
        P1_point = geo.addPoint(P1[0], P1[1], 0)
        P2_point = geo.addPoint(P2[0], P2[1], 0)
        P3_point = geo.addPoint(P3[0], P3[1], 0)

        control_line_lwr = geo.addLine(P0_point, P1_point)
        control_line_lwr = geo.addLine(P2_point, P3_point)

    logging.info(f"Writing GMSH .geo file with filename = {project_id}.geo_unrolled")

    geo.addCurveLoop(list(range(aerofoil_start, aerofoil_end)), 1)
    geo.addPlaneSurface([1], 1)

    gmsh.write(f"{project_id}.geo_unrolled")
    gmsh.finalize()

    return

    #region ==== ADD DOMAIN POINTS ==== #

    # Refinement zone
    leading_edge = aerofoil_points[int(len(aerofoil_points)/2)+1]
    trailing_edge = aerofoil_points[0]
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