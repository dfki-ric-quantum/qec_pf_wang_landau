import os
import random
import shutil
import subprocess
import sys
import time
from multiprocessing import Pool

import h5py
import numpy as np


def check_interactions(interactions, prob, tol=0.02):
    check_passed = True

    count = np.count_nonzero(interactions == -1)
    ratio = count / (interactions.shape[0] * interactions.shape[1])

    if abs(ratio - prob) >= tol:
        print("Interactions don't fulfill tolerance criteria")
        check_passed = False

    return check_passed

def run_test(dimX, dimY, prob, num_iterations, num_walker, num_intervals, num_interactions, task_id, boundary):
    seed = int(time.time())

    arguments = [
        "-x",
        str(dimX),
        "-y",
        str(dimY),
        "-p",
        str(prob),
        "-n",
        str(num_iterations),
        "-w",
        str(num_walker),
        "-s",
        str(seed),
        "-i",
        str(num_intervals),
        "-r",
        str(num_interactions),
        "-d",
        str(task_id),
        "--boundary",
        str(boundary),
    ]
    subprocess.run( [binary_path] + arguments, capture_output=True, text=True )

    boundary_type = "periodic" if boundary == 0 else "open"
    file = h5py.File(
        f"init/task_id_{task_id}/seed_{seed}/{boundary_type}/prob_{prob:.6f}/X_{dimX}_Y_{dimY}/error_class_i/prerun_results.h5",
        "r",
    )

    seeds = np.arange(num_interactions)

    input = [
        (
            np.array(file[f"//Interaction/{s}/Interaction"]).reshape((2 * dimX, dimY)),
            prob,
        )
        for s in seeds
    ]

    with Pool(128) as pool:
        results = pool.starmap(check_interactions, input)

    file.close()
    shutil.rmtree("init")

    return all(results)

"""In this test, we check whether the flipped interaction ratio is in line with the given probability."""
if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Wrong command line parameters")
        sys.exit(-1)

    binary_path = sys.argv[1] + "/build/src/prerun"

    if not os.path.isfile(binary_path) or not os.access(binary_path, os.X_OK):
        print("Prerun binary does not exist/can't be executed'")
        sys.exit(-1)

    dimX = 100
    dimY = 100
    num_iterations = 10
    num_intervals = 60
    num_walker = 128
    num_interactions = 1000
    prob = round(random.uniform(0, 1), 4)
    task_id = 99

    periodic_passed = run_test(dimX, dimY, prob, num_iterations, num_walker, num_intervals, num_interactions, task_id, 0)
    open_passed = run_test(dimX, dimY, prob, num_iterations, num_walker, num_intervals, num_interactions, task_id, 1)


    if (periodic_passed and open_passed):
        sys.exit(0)
    else:
        sys.exit(-1)
