import json
import os

def generate_fluent_journal(parameters:dict, commands:dict):
	journal = []
	
	def initialise_solver(journal:list):
		journal.extend(commands["set_model"])
		journal.extend(commands["set_methods"])
		journal.extend(commands["create_cl_report"])
		journal.extend(commands["create_cd_report"])
		journal.extend(commands["set_residuals_thresholds"])
		
		return journal
	
	journal = initialise_solver(journal)
	
	# Simple test
	journal.extend(commands["read_mesh"])
	journal.extend(commands["set_inlet"])
	journal.extend(commands["update_reference_values"])
	journal.extend(commands["initialise_solution"])
	journal.extend(commands["run_solution"])
	
	return journal
	
if __name__ == "__main__":
	with open("parameters.json", "r") as f:
		parameters = json.load(f)
		
	with open("scripts/fluent-definitions.json", "r") as f:
		commands = json.load(f)

	journal = generate_fluent_journal(parameters, commands)
	
	with open("scripts/journal.jou", "w", newline="\n") as f:
		f.writelines(journal)