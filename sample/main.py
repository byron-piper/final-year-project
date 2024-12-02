import os
import sys
import traceback
from datetime import datetime
import logging

from gmsher import construct_gmsh
from aerofoil_gen import generate_naca4

def log_uncaught_exceptions(exc_type, exc_value, exc_tb):
    if exc_value != KeyboardInterrupt:
        logging.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_tb))
    
    sys.__excepthook__(exc_type, exc_value, exc_tb)

def main():
    logging.info('Constructing the GMSH .geo file.')
    construct_gmsh()

if __name__ == '__main__':
    sys.excepthook = log_uncaught_exceptions

    if not os.path.exists("logs"):
        os.mkdir("logs")

    logging.basicConfig(
        filename=f"logs/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log",
        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S',
        level=logging.DEBUG
    )

    main()   