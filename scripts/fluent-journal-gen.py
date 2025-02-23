import glob
import json
import os
import math
import shutil

def set_inlet(command:list, freestream_velocity:float, aoa:float):
	command = command[:]
	velocity_x = freestream_velocity * math.cos(math.radians(aoa))
	velocity_y = freestream_velocity * math.sin(math.radians(aoa))

	command[24] = command[24].format(velocity_x)
	command[25] = command[25].format(velocity_y)

	return command

def read_mesh(command:list, filepath:str):
	command = command[:]
	command[1] = command[1].format(filepath)
	return command

def run_solution(command:list, iterations:int):
	command = command[:]
	command[4] = command[4].format(iterations)
	return command

def export_solution_data(command:list, filepath:str):
	command = command[:]
	command[8] = command[8].format(filepath)
	return command

def calc_freestream_velocity(x:float, Re:float, rho:float, mu:float):
	return (Re * mu) / (x * rho)

def generate_fluent_journal(parameters:dict, commands:dict):
	def initialise_solver(journal:list):
		journal.extend(commands["set_model"])
		journal.extend(commands["set_methods"])
		journal.extend(commands["set_residuals_thresholds"])
		journal.append("\n")
		
		return journal
	
	journal = []

	#region # ==== UNPACK PARAMETERS ==== #

	chord = parameters["coords"]["chord"]
	freestream_reynolds = parameters["solver"]["freestream_reynolds"]
	freestream_density = parameters["solver"]["freestream_density"]
	freestream_viscosity = parameters["solver"]["freestream_viscosity"]
	iterations = parameters["solver"]["iterations"]
	meshes_folder = parameters["i/o"]["meshes_folder"]
	results_folder = parameters["i/o"]["results_folder"]
	overwrite_results = parameters["solver"]["overwrite_results"]

	#endregion

	freestream_velocity = calc_freestream_velocity(chord, freestream_reynolds, freestream_density, freestream_viscosity)

	meshes = []
	for file in glob.glob(os.path.join(meshes_folder, "*.msh.h5")):
		filepath = os.path.abspath(file)
		mesh_name = os.path.basename(file).split('.')[0]
		meshes.append((filepath, mesh_name))

	#journal = initialise_solver(journal)

	# Delete old results if overwrite set to true
	if overwrite_results:
		for item in os.listdir(results_folder):
			item_path = os.path.join(results_folder, item)
			if os.path.isfile(item_path) or os.path.islink(item_path):
				os.remove(item_path)
			elif os.path.isdir(item_path):
				shutil.rmtree(item_path)

	for i, mesh in enumerate(meshes):
		mesh_filepath = mesh[0]
		mesh_name = mesh[1]

		# Create results folder for mesh
		results_folder = os.path.join(results_folder, mesh_name)
		if not os.path.exists(results_folder):
			os.mkdir(results_folder)

		# LINEAR SOLUTIONS
		journal.extend(read_mesh(commands["read_mesh"], mesh_filepath))

		if i == 0:
			journal = initialise_solver(journal)

		for aoa in [-5, -2.5]:
			solution_data_filepath = os.path.join(
				results_folder,
				f"{mesh_name}_LIN_a{aoa}.csv"
			)

			journal.extend(set_inlet(commands["set_inlet"], freestream_velocity, aoa))
			#print(commands["set_inlet"])


			journal.extend(commands["update_reference_values"])
			journal.extend(commands["initialise_solution"])
			journal.extend(run_solution(commands["run_solution"], iterations))
			journal.extend(export_solution_data(commands["export_solution_data"], solution_data_filepath))
			journal.append("\n")
		break

	return journal
	
if __name__ == "__main__":
	with open("parameters.json", "r") as f:
		parameters = json.load(f)
		
	with open("scripts/fluent-definitions.json", "r") as f:
		commands = json.load(f)

	journal = generate_fluent_journal(parameters, commands)
	
	with open("scripts/journal.jou", "w") as f:
		f.writelines(journal)