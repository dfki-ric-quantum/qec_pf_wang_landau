import os
import random
import shutil
import subprocess
import sys
import time
from multiprocessing import Pool

import h5py
import numpy as np


def calc_periodic_energy(lattice, interactions):
    energy = 0.0
    nx = lattice.shape[0]
    for i in range(lattice.shape[0]):
        for j in range(lattice.shape[1]):
            inn = i - 1 if (i - 1 >= 0) else lattice.shape[0] - 1
            jnn = j - 1 if (j - 1 >= 0) else lattice.shape[1] - 1
            energy += lattice[i, j] * ( lattice[inn, j] * interactions[nx + inn, j] + lattice[i, jnn] * interactions[i, jnn] )
    return -energy

def calc_open_energy(lattice, interactions):
    energy = 0.0
    nx = lattice.shape[0]
    for i in range(lattice.shape[0]):
        for j in range(lattice.shape[1]):
            inn = i - 1
            c_i = 1 if (i - 1 >= 0) else 0
            jnn = j - 1
            c_j = 1 if (j - 1 >= 0) else 0
            energy += lattice[i, j] * ( c_i * lattice[inn, j] * interactions[nx + inn, j] + c_j * lattice[i, jnn] * interactions[i, jnn] )
    return -energy



def check_energies(interaction_id, path, dimX, dimY, boundary):
    check_passed = True

    file = h5py.File(path, "r")

    interaction = np.array(file[f"//Interaction/{interaction_id}/Interaction"]).reshape(
        2 * dimX, dimY
    )

    energies = list(file[f"//Lattice/{interaction_id}"].keys())

    check_energies = random.sample(energies, int(len(energies) * 0.5))

    for en in check_energies:
        lattice = np.array(file[f"//Lattice/{interaction_id}/{en}/Lattice"]).reshape(
            dimX, dimY
        )

        if boundary == 0: # periodic - 0
            py_energy = calc_periodic_energy(lattice, interaction)
        elif boundary == 1: # open - 1
            py_energy = calc_open_energy(lattice, interaction)
        else:
            print("Unallowed boundary handed")
            return False

        if py_energy != int(en):
            print("Python energy ", py_energy, " and C++ ", en)
            check_passed = False

    file.close()

    return check_passed

def run_test(binary_path, boundary, dimX, dimY, prob, num_iterations, num_walker, num_intervals, num_interactions, task_id):
    seed = int(time.time())

    arguments = [
        "-x", str(dimX), "-y", str(dimY), "-p", str(prob),
        "-n", str(num_iterations), "-w", str(num_walker),
        "-s", str(seed), "-i", str(num_intervals),
        "-r", str(num_interactions), "-d", str(task_id),
        "--boundary", str(boundary),
    ]
    subprocess.run([binary_path] + arguments, capture_output=True, text=True)

    boundary_type = "periodic" if boundary == 0 else "open"
    path = f"init/task_id_{task_id}/seed_{seed}/{boundary_type}/prob_{prob:.6f}/X_{dimX}_Y_{dimY}/error_class_i/prerun_results.h5"

    input_data = [(i, path, dimX, dimY, boundary) for i in range(num_interactions)]
    with Pool(num_interactions) as pool:
        results = pool.starmap(check_energies, input_data)

    shutil.rmtree("init")

    return all(results)


"""In this test we are checking whether the energies written in the hdf5 data matches the actual energy
by comparing it with Python calculated energies."""
if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Wrong command line parameters")
        sys.exit(-1)

    binary_path = sys.argv[1] + "/build/src/prerun"

    if not os.path.isfile(binary_path) or not os.access(binary_path, os.X_OK):
        print("Prerun binary does not exist/can't be executed'")
        sys.exit(-1)

    dimX = 12
    dimY = 12
    prob = round(random.uniform(0, 1), 4)
    num_iterations = 10000
    num_intervals = 40
    num_walker = 128
    num_interactions = 100
    task_id = 99

    periodic_passed = run_test(
        binary_path, boundary=0, dimX=dimX, dimY=dimY, prob=prob,
        num_iterations=num_iterations, num_walker=num_walker,
        num_intervals=num_intervals, num_interactions=num_interactions, task_id=task_id
    )

    open_passed = run_test(
        binary_path, boundary=1, dimX=dimX, dimY=dimY, prob=prob,
        num_iterations=num_iterations, num_walker=num_walker,
        num_intervals=num_intervals, num_interactions=num_interactions, task_id=task_id
    )

    if (periodic_passed and open_passed):
        sys.exit(0)
    else:
        sys.exit(-1)
