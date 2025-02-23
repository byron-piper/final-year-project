project_id = "dev"
log_enabled = True
debug_enabled = False
randomise = False

#region ==== AEROFOIL ==== #

chord = 1

aerofoil_template = [
    {
        "id": "base",
        "chord": 1,
        "params": {
            "chord": 1,
            "aoa": 0,
            "naca": [2, 4, 12],
            "num_points": 200
        },
        "coords": []
    },
    {
        "id": "slat",
        "chord": 1,
        "params": {
            "Tx": 0.15,
            "Ux": 0.1,
            "Vx": 0.5,
            "Wx": 0.5,
            "x_offset": 0.01,
            "slot_gap": 0.025,
            "deflection": 20
        },
        "coords": []
    },
    {
        "id": "flap",
        "chord": 1,
        "params": {
            "Bx": 0.7,
            "By": 0.25,
            "Cx": 0.82,
            "Fx": 0.75,
            "My": 0.5,
            "Nx": 0.5,
            "Ly": 0.8,
            "Gx": 0.4,
            "Px": 0.25,
            "Sx": 0.65,
            "P1x": 0.3,
            "S1x": 0.25,
            "x_offset": -0.02,
            "slot_gap": 0.02,
            "deflection": -30
        },
        "coords": []
    }
]

#endregion

#region ==== HIGH LIFT DEVICES ==== #

slat_geom = {
    "Tx": 0.15,
    "Ux": 0.1,
    "Vx": 0.6,
    "Wx": 0.25,
    "x_offset": 0.01,
    "slot_gap": 0.025,
    "deflection": 20
}

flap_geom = {
    "Bx": 0.7,
    "By": 0.25,
    "Cx": 0.82,
    "Fx": 0.75,
    "My": 0.5,
    "Nx": 0.5,
    "Ly": 0.8,
    "Gx": 0.4,
    "Px": 0.25,
    "Sx": 0.65,
    "P1x": 0.3,
    "S1x": 0.25,
    "x_offset": -0.02,
    "slot_gap": 0.02,
    "deflection": -30
}

#endregion

#region ==== GMSHER ==== #

mesh_size = 0.2
domain_radius = 7
domain_extension = 15
AoA = 35
N_inlet = 100
N_vertical = 30
P_vertical = 1.1
N_shear = 50
P_shear = 1.01
refinement_mult = 0.35
wake_mult = 2
y_plus = 0.5

#endregion

#region # ==== SOLVER ==== #

freestream_velocity = 30
freestream_density = 1.225
dynamic_viscosity = 3.178e-5

solver_inputs = [freestream_velocity, freestream_density, dynamic_viscosity]

#endregion