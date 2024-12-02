import os
import sys
import traceback
from datetime import datetime
import logging
import random

import parameters
from gmsher import construct_gmsh
from aerofoil_gen import generate_naca4

def log_uncaught_exceptions(exc_type, exc_value, exc_tb):
    if exc_value != KeyboardInterrupt:
        logging.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_tb))
    
    sys.__excepthook__(exc_type, exc_value, exc_tb)


def randomise_parameters():
    parameters.slat_geom = {
        "Tx": round(random.uniform(0.05, 0.3), 5),
        "Ux": round(random.uniform(0.01, 1), 5),
        "Vx": round(random.uniform(0.01, 1), 5),
        "Wx": round(random.uniform(0.01, 1), 5),
        "x_overlap": round(random.uniform(0.01, 0.1), 5),
        "y_gap": round(random.uniform(0.02, 0.1), 5),
        "deflection": round(random.uniform(0, 45), 5)
    }
    parameters.flap_geom = {
        "Bx": round(random.uniform(0.5, 0.9), 5),
        "By": round(random.uniform(0.05, 0.95), 5),
        "Cx": round(random.uniform(0.05, 0.95), 5),
        "Fx": round(random.uniform(0.05, 0.95), 5),
        "My": round(random.uniform(0.05, 0.95), 5),
        "Nx": round(random.uniform(0.05, 0.95), 5),
        "Ly": round(random.uniform(0.05, 0.95), 5),
        "Gx": round(random.uniform(0.05, 0.95), 5),
        "Px": round(random.uniform(0.05, 0.95), 5),
        "Sx": round(random.uniform(0.05, 0.95), 5),
        "P1x": round(random.uniform(0.05, 0.95), 5),
        "S1x": round(random.uniform(0.05, 0.95), 5),
        "x_overlap": round(random.uniform(0.01, 0.1), 5),
        "y_gap": round(random.uniform(0.02, 0.1), 5),
        "deflection": round(-random.uniform(0, 45), 5)
    }  


def main():
    if parameters.randomise:
        logging.info('Randomising parameters')
        randomise_parameters()
        logging.info(f'Updated parameters:\nSlat geom = {parameters.slat_geom}\nFlap geom = {parameters.flap_geom}')

    logging.info('Constructing the GMSH .geo file.')
    construct_gmsh()

if __name__ == '__main__':
    sys.excepthook = log_uncaught_exceptions

    if not os.path.exists("logs"):
        os.mkdir("logs")

    logging.basicConfig(
        filename=f"logs/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log",
        format='%(asctime)s | %(levelname)s | %(filename)s : %(message)s',
        datefmt='%H:%M:%S',
        level=logging.DEBUG
    )

    main()   