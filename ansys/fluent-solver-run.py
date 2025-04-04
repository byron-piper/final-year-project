from datetime import datetime, timedelta
import glob
import logging
import math
import os
import sys

import ansys.fluent.core as pyfluent
import numpy as np

project_path = r"C:\Users\honey\Documents\PROJECT\final-year-project-main"
sys.path.append(project_path)

from sample.helper import load_parameters, calc_freestream_velocity

def configure_solver_settings(solver, iterations:int):
    #region docstring
    """
    Configures the solver settings to default values:
        - Viscous model = `spalart-allmaras`
        - Velocity specification method = `Components`
        - Residuals absolute criteria [continuity, x-velocity, y-velocity, nut] = `1e-6`
        - Iterations = `iterations`
    
    Parameters
    ----------
    None
        
    Returns
    -------
    unknown
        Fluent inlent object
    """
    #endregion
    
    logging.info("[-/-] : - | Running setup...")
    solver.settings.setup.models.viscous.model = 'spalart-allmaras'

    inlet = solver.settings.setup.boundary_conditions.velocity_inlet['inlet']
    inlet.momentum.velocity_specification_method = "Components"

    solver.settings.solution.monitor.residual.equations['continuity'].absolute_criteria = 1e-6
    solver.settings.solution.monitor.residual.equations['x-velocity'].absolute_criteria = 1e-6
    solver.settings.solution.monitor.residual.equations['y-velocity'].absolute_criteria = 1e-6
    solver.settings.solution.monitor.residual.equations['nut'].absolute_criteria = 1e-6

    solver.settings.solution.run_calculation.iter_count = iterations

    logging.info("[-/-] : - | Setup complete.")

    return inlet

def run_solver(params:dict) -> None:
    #region
    """
    Performs a 2D Fluent simulation for each `.h5` mesh file found in the
    meshes folder given by `parameters`. Each simulation is ran under the given
    conditions:
        - Spalart-Allmaras viscous model
        - 15 Angle of Attacks ranging from {-2.5, 15} degrees by manipulating velocity components
        - 1e-6 absolute criteria for each residual
    
    Saves flowfield data in `.csv` files and lift- and drag-coefficients in `.out` files
    
    Parameters
    ----------
    params : dict
        Dictionary containing workflow parameters
        
    Returns
    -------
    None
    """
    #endregion
    
    #region # ====  UNPACK PARAMETERS ==== #

    chord = params["coords"]["chord"]

    freestream_reynolds = params["solver"]["freestream_reynolds"]
    freestream_density = params["solver"]["freestream_density"]
    freestream_viscosity = params["solver"]["freestream_viscosity"]
    iterations = params["solver"]["iterations"]
    
    meshes_folder = os.path.join(params["i/o"]["meshes_folder"], "study")
    cases_folder = params["i/o"]["cases_folder"]
    results_folder = os.path.join(params["i/o"]["results_folder"], "study")
    logs_folder = os.path.join(params["i/o"]["logs_folder"], "solver")

    #endregion
    
    if not os.path.exists(logs_folder):
        os.mkdir(logs_folder)

    # Initialise logging
    log_filename = os.path.join(logs_folder,
                                datetime.now().strftime("log_%Y-%m-%d_%H-%M-%S.log"))

    # Configure logging
    logging.basicConfig(filename=log_filename, level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")

    freestream_velocity = calc_freestream_velocity(freestream_reynolds, chord, freestream_density, freestream_viscosity)
    logging.info(f"[-/-] : - | Freestream velocity = '{freestream_velocity}'")

    angle_of_attacks = np.linspace(-5, 30, 15)
    
    angle_of_attacks = [
        np.linspace(5, 7.5, 10),
        np.linspace(10, 12.5, 10),
        np.linspace(10, 12.5, 10)
    ]
    
    # Walk through each subfolder within the import location and look for three '.txt. files containing
    # coordinates for each aerofoil element. Save the name of the folder and its absolute path
    meshes = []
    for file in glob.glob(os.path.join(meshes_folder, "*.msh.h5")):
        meshes.append((file, os.path.basename(file).split('.')[0]))

    logging.info("[-/-] : - | Initialising Fluent solver...")
    # Initialise solver
    solver = pyfluent.launch_fluent(dimension=2, precision='double', processor_count=4, mode='solver', show_gui=True)

    cumulative_time = 0
    completed_loops = 1
    for i, mesh in enumerate(meshes):
        try:
            mesh_filename = mesh[0]
            mesh_name = mesh[1]
            mesh_results_folder = os.path.join(results_folder, mesh_name)
            if not os.path.exists(mesh_results_folder):
                os.mkdir(mesh_results_folder)
            mesh_cases_folder = os.path.join(cases_folder, mesh_name)
            if not os.path.exists(mesh_cases_folder):
                os.mkdir(mesh_cases_folder)
            
            logging.info(f"[-/-] : - | Reading in mesh '{mesh_name}'...")
            solver.settings.file.read_mesh(file_name = mesh_filename)
            logging.info("[-/-] : - | Mesh imported successfully.")

            inlet = configure_solver_settings(solver, iterations)

            print(angle_of_attacks[i])
            for j, aoa in enumerate(angle_of_attacks[i]):
                start_time = datetime.now()
                progress = f"[{completed_loops}/{len(angle_of_attacks[i])*len(meshes)}]"
                mesh_results_filename = os.path.join(mesh_results_folder, f"{str(aoa)}.csv")
                mesh_cl_filename = os.path.join(mesh_results_folder, f"cl_{str(aoa)}.out")
                mesh_cd_filename = os.path.join(mesh_results_folder, f"cd_{str(aoa)}.out")
                mesh_case_filename = os.path.join(mesh_cases_folder, f"{str(aoa)}.cas.h5")

                logging.info(f"{progress} : {mesh_name} | Angle of attack = '{aoa}'")

                logging.info(f"{progress} : {mesh_name} | Creating report definitions...")
                solver.settings.solution.report_definitions.lift["cl"] = {
                    "zones": ["part-boundary"],
                    "force_vector": [math.sin(math.radians(aoa)), math.cos(math.radians(aoa)), 1]
                }
                solver.settings.solution.report_definitions.lift["cd"] = {
                    "zones": ["part-boundary"],
                    "force_vector": [math.cos(math.radians(aoa)), math.sin(math.radians(aoa)), 1]
                }
                solver.settings.solution.monitor.report_files["cl-rfile"] = {
                    "report_defs": ["cl"],
                    "file_name": mesh_cl_filename
                }
                solver.settings.solution.monitor.report_files["cd-rfile"] = {
                    "report_defs": ["cd"],
                    "file_name": mesh_cd_filename
                }

                logging.info(f"{progress} : {mesh_name} | Calculating velocity components...")
                velocity_x = freestream_velocity * math.cos(math.radians(aoa))
                velocity_y = freestream_velocity * math.sin(math.radians(aoa))

                logging.info(f"{progress} : {mesh_name} | Updating inlet boundary...")
                inlet.momentum.velocity_components[0] = velocity_x
                inlet.momentum.velocity_components[1] = velocity_y

                logging.info(f"{progress} : {mesh_name} | Updating reference values...")
                solver.settings.setup.reference_values.velocity = math.sqrt(math.pow(velocity_x, 2) + math.pow(velocity_y, 2))
                solver.settings.setup.reference_values.zone = "fluid"

                logging.info(f"{progress} : {mesh_name} | Performing standard initialisation...")
                solver.settings.solution.initialization.defaults['x-velocity'] = velocity_x
                solver.settings.solution.initialization.defaults['y-velocity'] = velocity_y
                solver.settings.solution.initialization.defaults['nut'] = 0.0001460735
                solver.settings.solution.initialization.standard_initialize()

                logging.info(f"{progress} : {mesh_name} | Beginning calculation...")
                solver.settings.solution.run_calculation.calculate()

                logging.info(f"{progress} : {mesh_name} | Saving results to '{mesh_results_filename}'...")
                solver.settings.file.export.ascii(
                    file_name=mesh_results_filename,
                    surface_name_list=["interior:fluid"],
                    cell_func_domain=[
                        "pressure-coefficient",
                        "velocity-magnitude",
                        "vorticity-mag"
                    ]
                )

                logging.info(f"{progress} : {mesh_name} | Saving case file to '{mesh_case_filename}'...")
                solver.settings.file.write_case(file_name=mesh_case_filename)

                end_time = datetime.now()
                time_elapsed = end_time - start_time
                cumulative_time += time_elapsed.seconds
                meshes_remaining = (len(angle_of_attacks[i])*len(meshes)) - (completed_loops)
                avg_time_per_iteration = cumulative_time / (completed_loops)
                estimated_time_remaining = timedelta(seconds=(avg_time_per_iteration * meshes_remaining))

                logging.info(f"{progress} : {mesh_name} | Time elapsed = {time_elapsed}")
                logging.info(f"{progress} : {mesh_name} | Estimated time remaining = {estimated_time_remaining}")
                completed_loops += 1
        except Exception as e:
            logging.error(f"{progress} : {mesh_name} | Error occured during solving procress: '{e}'. Skipping!")
                    
if __name__ == "__main__":
    parameters = load_parameters()

    run_solver(parameters)
