import glob
import gmsh
import os
import random
import shutil

import numpy as np

from aerofoil_utils import generate_naca4, generate_flap_coords, generate_slat_coords, normalise_aerofoil_coords, save_highlift_aerofoil, aerofoil_elements_intersecting, visualise_aerofoil_coords
from helper import load_parameters

def generate_random_slat_params(slat_batch_size:int):
    slat_params = {}

    attempts = 0
    while(len(slat_params) != slat_batch_size):
        attempts += 1
        if attempts > 1000: break

        # Parameters defining shape and positioning of slat element
        Tx = round(random.uniform(0.2, 0.3), 2)
        Ux = round(random.uniform(0.05, 0.1), 2)
        x_offset = round(random.uniform(0, 0.025), 3)
        y_offset = round(random.uniform(0.025, 0.05), 3)
        deflection = round(random.uniform(0, 40))

        # Generate unique identifier for slat parameters combination
        slat_id = "".join([
            f"{str(int(Tx*100)).zfill(2)}",
            f"{str(int(Ux*100)).zfill(2)}",
            f"{str(int(x_offset*1000)).zfill(2)}",
            f"{str(int(y_offset*1000)).zfill(2)}",
            f"{str(int(deflection)).zfill(2)}",
        ])

        if slat_id in slat_params: 
            print("No more unique parameters found!")
            continue

        slat_params[slat_id] = {
            "Tx": Tx,
            "Ux": Ux,
            "Vx": 0.5,
            "Wx": 0.5,
            "x_offset": x_offset,
            "y_offset": y_offset,
            "deflection": deflection
        }

    return slat_params

def generate_random_flap_params(flap_batch_size:int):
    flap_params = {}

    attempt = 0
    while(len(flap_params) != flap_batch_size):
        attempt += 1
        if attempt > 1000:
            print("No more unique parameters found!")
            break

        # Parameters defining shape and positioning of flap element
        Bx = round(random.uniform(0.5, 0.8), 2)
        Cx = round(random.uniform(0.5, 0.75), 2)
        Fx = round(random.uniform(0.5, 0.75), 2)
        Sx = round(random.uniform(0.5, Bx-0.1), 2)
        x_offset = round(random.uniform(0.00, 0.025), 3)
        y_offset = round(random.uniform(0.025, 0.05), 3)
        deflection = round(random.uniform(0, 40))

        # Generate unique identifier for flap parameters combination
        flap_id = "".join([
            f"{str(int(Bx*100)).zfill(2)}",
            f"{str(int(Cx*100)).zfill(2)}",
            f"{str(int(Fx*100)).zfill(2)}",
            f"{str(int(Fx*100)).zfill(2)}",
            f"{str(int(x_offset*1000)).zfill(2)}",
            f"{str(int(y_offset*1000)).zfill(2)}",
            f"{str(int(deflection)).zfill(2)}",
        ])

        if flap_id in flap_params: 
            print("No more unique parameters found!")
            continue

        flap_params[flap_id] = {
            "Bx": Bx,
            "By": 0.25,
            "Cx": Cx,
            "Fx": Fx,
            "My": 0.5,
            "Nx": 0.5,
            "Ly": 0.8,
            "Gx": 0.4,
            "Px": 0.25,
            "Sx": Sx,
            "P1x": 0.3,
            "S1x": 0.25,
            "x_offset": x_offset,
            "y_offset": y_offset,
            "deflection": deflection
        }

    return flap_params

def generate_random_nacas(naca_batch_size:int):
    nacas = {}

    attempts = 0
    while(len(nacas) != naca_batch_size):
        attempts += 1
        if attempts > 1000: break

        # Generate random NACA-4 digits
        M, P, T = random.randint(0, 4), random.randint(2, 5), random.randint(10, 25)
        # Generate NACA ID
        naca_id = f"NACA-{M}{P}{str(T).zfill(2)}"

        # Skip aerofoil if this NACA configuration already exists
        if naca_id in nacas: continue

        # Generate random NACA-4 aerofoil coordinates
        nacas[naca_id] = generate_naca4(
            chord = 1,
            M = M,
            P = P,
            T = T,
            num_points = 200
        )

    return nacas

def generate_training_data(naca_batch_size:int, flap_batch_size:int, slat_batch_size:int):
    aerofoils = {}

    # Generate random NACA-4 aerofoils
    nacas = generate_random_nacas(naca_batch_size)
    # Generate random slat parameters
    slat_params_batch = generate_random_slat_params(slat_batch_size)
    # Generate random flap parameters
    flap_params_batch = generate_random_flap_params(flap_batch_size)

    for naca_id, naca_coords in nacas.items():
        for slat_id, slat_params in slat_params_batch.items():
            for flap_id, flap_params in flap_params_batch.items():
                slat_coords, base_coords = generate_slat_coords(1, naca_coords, slat_params)
                flap_coords, base_coords = generate_flap_coords(1, base_coords, flap_params)
                combined_key = f"{naca_id}-{flap_id}-{slat_id}"

                if aerofoil_elements_intersecting(base_coords, flap_coords, slat_coords):
                    print(f"Aerofoil '{combined_key}' has intersecting elements and will not be added to dataset.")
                    continue
                
                base_coords, flap_coords, slat_coords = normalise_aerofoil_coords([base_coords, flap_coords, slat_coords], 1)

                aerofoils[combined_key] = [base_coords, flap_coords, slat_coords]


    return aerofoils

def test():
    M, P, T = random.randint(0, 4), random.randint(1, 5), random.randint(10, 25)
    base_coords = generate_naca4(
            chord = 1,
            M = M,
            P = P,
            T = T,
            num_points = 200
    )

    slat_params = {
        "Tx": 0.25,
        "Ux": 0.1,
        "Vx": 0.5,
        "Wx": 0.5,
        "x_offset": 0.01,
        "y_offset": 0.025,
        "deflection": 30
    }

    flap_params = {
        "Bx": 0.7,
        "By": 0.25,
        "Cx": 0.8,
        "Fx": 0.6,
        "My": 0.5,
        "Nx": 0.5,
        "Ly": 0.8,
        "Gx": 0.4,
        "Px": 0.25,
        "Sx": 0.5,
        "P1x": 0.5,
        "S1x": 0.5,
        "x_offset": 0.01,
        "y_offset": 0.025,
        "deflection": 30
    }

    flap_params = list(generate_random_flap_params(1).values())[0]
    slat_params = list(generate_random_slat_params(1).values())[0]

    slat_coords, base_coords = generate_slat_coords(1, base_coords, slat_params)
    flap_coords, base_coords = generate_flap_coords(1, base_coords, flap_params)

    gmsh.initialize()

    #np.savetxt("output.txt", slat_coords, fmt="%f", delimiter=" ")

    #gmsh.model.geo.addSpline(p)

    for coord in base_coords:
        gmsh.model.geo.addPoint(coord[0], coord[1], 0, 1)
    for coord in flap_coords:
        gmsh.model.geo.addPoint(coord[0], coord[1], 0, 1)
    for coord in slat_coords:
        gmsh.model.geo.addPoint(coord[0], coord[1], 0, 1)

    gmsh.model.geo.synchronize()

    gmsh.write("test.geo_unrolled")
    gmsh.finalize()

    return slat_coords

if __name__ == "__main__":
    params = load_parameters()
    
    #region # ==== UNPACK PARAMETERS ==== #
    
    rn_seed = params["datagen"]["rn_seed"]
    num_nacas = params["datagen"]["num_nacas"]
    num_flaps = params["datagen"]["num_flaps"]
    num_slats = params["datagen"]["num_slats"]
    
    #endregion

    random.seed(rn_seed)

    aerofoils = generate_training_data(num_nacas, num_flaps, num_slats)

    columns = num_slats * num_flaps

    visualise_aerofoil_coords(aerofoils, columns=columns)

    folders = [f for f in glob.glob(os.path.join(params["i/o"]["coords_folder"], "*")) if os.path.isdir(f)]
    
    # Delete each folder found
    for folder in folders:
        try:
            shutil.rmtree(folder)
        except Exception:
            print(f"Cannot delete '{folder}'!")
            pass

    # Write the randomly generated coordinates to coordinates folder
    if params["coords"]["write_coords"]:
        for aerofoil_id, aerofoil_coords in aerofoils.items():
            save_highlift_aerofoil(aerofoil_id, aerofoil_coords, params["i/o"]["coords_folder"])

    