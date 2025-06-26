import itertools
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
    energy = 0

    nx = lattice.shape[0]

    for i in range(lattice.shape[0]):
        for j in range(lattice.shape[1]):
            inn = i - 1 if (i - 1 >= 0) else lattice.shape[0] - 1
            jnn = j - 1 if (j - 1 >= 0) else lattice.shape[1] - 1

            energy += lattice[i, j] * (
                lattice[inn, j] * interactions[nx + inn, j]
                + lattice[i, jnn] * interactions[i, jnn]
            )

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

def run_test(binary_path, boundary, dimX, dimY, prob, num_iterations, num_walker, num_intervals, num_interactions, task_id):
    seed = int(time.time())
    arguments = [
        "-x", str(dimX), "-y", str(dimY), "-p", str(prob),
        "-n", str(num_iterations), "-w", str(num_walker),
        "-s", str(seed), "-i", str(num_intervals),
        "-r", str(num_interactions), "-d", str(task_id),
        "--boundary", str(boundary),
    ]

    subprocess.run(
        [binary_path] + arguments,
        capture_output=True,
        text=True,
    )

    boundary_type = "open" if boundary else "periodic"
    file = h5py.File(
        f"init/task_id_{task_id}/seed_{seed}/{boundary_type}/prob_{prob:.6f}/X_{dimX}_Y_{dimY}/error_class_i/prerun_results.h5",
        "r",
    )

    ## Get all possible spin configurations for a 4x4 lattice
    num_spins = dimX * dimY
    lattices = np.array(list(itertools.product([-1, 1], repeat=num_spins)))
    lattices = lattices.reshape(-1, dimX, dimY)

    Emin = -2 * dimX * dimY

    energies_match = []

    for i in range(num_interactions):
        # Read in histogram
        energies_flag = np.array(file[f"//Histogram/{i}/Histogram"])

        # Find all found energies
        indices = np.where(energies_flag == 1)[0]
        energies_hist = set(indices + Emin)

        # Read in interactions
        interactions = np.array(file[f"//Interaction/{i}/Interaction"]).reshape(
            (2 * dimX, dimY)
        )

        input = tuple(zip(lattices, [interactions] * len(lattices)))

        with Pool(128) as pool:
            if boundary:
                energies = pool.starmap(calc_open_energy, input)
            else:
                energies = pool.starmap(calc_periodic_energy, input)

        energies = set(energies)

        energies_match.append(energies == energies_hist)

    file.close()
    shutil.rmtree("init")
    return all(energies_match)

"""
In this test we are performing the prerun for lattice sizes 4x4 and generate histograms.
After this we are generating all possible combinations for the 4x4 lattice, calculate all
possible energies and compare them to the found energies in prerun. If they are the same,
the test is correct.
"""
if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Wrong command line parameters")
        sys.exit(-1)

    binary_path = sys.argv[1] + "/build/src/prerun"

    if not os.path.isfile(binary_path) or not os.access(binary_path, os.X_OK):
        print("Prerun binary does not exist/can't be executed'")
        sys.exit(-1)

    dimX = 4
    dimY = 4
    prob = round(random.uniform(0, 1))
    num_iterations = 10000
    num_intervals = 20
    num_walker = 128
    num_interactions = 10
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
