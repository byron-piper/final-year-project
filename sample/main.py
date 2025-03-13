from aerofoil_utils import generate_naca4, visualise_aerofoil_coords

coords = generate_naca4()

aerofoils = {
    "1": [generate_naca4(), generate_naca4(M=5)],
    "2": [generate_naca4()],
    "3": [generate_naca4()],
    "4": [generate_naca4()],
    "5": [generate_naca4()],
    "6": [generate_naca4()],
    "7": [generate_naca4()],
    "8": [generate_naca4()],
    "9": [generate_naca4()]
}

visualise_aerofoil_coords(aerofoils)