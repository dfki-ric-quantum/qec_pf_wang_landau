import os
import random
import shutil
import subprocess
import sys
import time
from multiprocessing import Pool

import h5py
import numpy as np

def log_sum_exp(to_sum):
    maxval = max(to_sum)
    exp_sum = 0
    for value in to_sum:
        exp_sum += np.e**(value-maxval)
    res = maxval + np.log(exp_sum)
    return res

def test(dimX, dimY, prob, num_iterations_prerun, num_iteration_mainrun,
         num_intervals_mainrun, walker_per_interval, alpha, beta, overlap,
         num_intervals_prerun, num_walker, num_interactions, replica_exchange_offsets,
         logical_error_type, seed_prerun, seed_mainrun, task_id, time_limit, boundary):
    # Arguments for call
    prerun_arguments = [
        "--X",
        str(dimX),
        "--Y",
        str(dimY),
        "--prob",
        str(prob),
        "--nit",
        str(num_iterations_prerun),
        "--nw",
        str(num_walker),
        "--seed",
        str(seed_prerun),
        "--num-intervals",
        str(num_intervals_prerun),
        "-e",
        logical_error_type,
        "--disorder-samples",
        str(num_interactions),
        "-d",
        str(task_id),
        "--boundary",
        str(boundary),
    ]

    mainrun_arguments = [
        "-x", str(dimX),
        "-y", str(dimY),
        "-n", str(num_iteration_mainrun),
        "-p", str(prob),
        "-a", str(alpha),
        "-b", str(beta),
        "-i", str(num_intervals_mainrun),
        "-w", str(walker_per_interval),
        "-o", str(overlap),
        "-s", str(seed_mainrun),
        "-e", logical_error_type,
        "-r", str(num_interactions),
        "-c", str(replica_exchange_offsets),
        "-d", str(task_id),
        "--time_limit", str(time_limit),
        "--boundary", str(boundary),
    ]

    Emin = -2 * dimX * dimY
    tolerance = 1e-12

    # call binaries
    result_prerun = subprocess.run(
        [prerun_binary_path] + prerun_arguments,
        capture_output=True,
        text=True,
    )
    # print("Prerun stdout:", result_prerun.stdout)
    # print("Prerun stderr:", result_prerun.stderr)

    # Run the command with subprocess.run
    result_mainrun = subprocess.run(
    [mainrun_binary_path] + mainrun_arguments,
    capture_output=True,
    text=True,
    )
    # print("Mainrun stdout:", result_mainrun.stdout)
    # print("Mainrun stderr:", result_mainrun.stderr)

    if(boundary):
        path_prerun = f"init/task_id_{task_id}/seed_{seed_prerun}/open/prob_{prob:.6f}/X_{dimX}_Y_{dimY}/error_class_{logical_error_type}/prerun_results.h5"
        path_mainrun = f"results/task_id_{task_id}/seed_{seed_mainrun}/open/prob_{prob:.6f}/X_{dimX}_Y_{dimY}/error_class_{logical_error_type}/mainrun_results.h5"
    else:
        path_prerun = f"init/task_id_{task_id}/seed_{seed_prerun}/periodic/prob_{prob:.6f}/X_{dimX}_Y_{dimY}/error_class_{logical_error_type}/prerun_results.h5"
        path_mainrun = f"results/task_id_{task_id}/seed_{seed_mainrun}/periodic/prob_{prob:.6f}/X_{dimX}_Y_{dimY}/error_class_{logical_error_type}/mainrun_results.h5"


    file_prerun = h5py.File(path_prerun, "r")
    file_mainrun = h5py.File(path_mainrun, "r")

    test_disorder_id = random.choice(list(file_mainrun[f"/"]))

    prerun_energies_flag = np.array(file_prerun[f"//Histogram/{test_disorder_id}/Histogram"])
    prerun_energy_indices = np.where(prerun_energies_flag == 1)[0]
    prerun_energies = set(prerun_energy_indices + Emin)

    test_result_group = random.choice(list(file_mainrun[f"/{test_disorder_id}/{seed_mainrun}"].keys()))

    mainrun_results = list(file_mainrun[f"/{test_disorder_id}/{seed_mainrun}/{test_result_group}/log_g"])

    mainrun_energies = {entry[0] for entry in mainrun_results}

    mainrun_log_g_values = [entry[1] for entry in mainrun_results]

    file_prerun.close()
    file_mainrun.close()

    return [mainrun_energies == prerun_energies, np.isclose(log_sum_exp(mainrun_log_g_values), np.log(2)*dimX*dimY, tolerance)], prerun_energies, mainrun_energies, mainrun_log_g_values, prerun_arguments, mainrun_arguments

"""In this test we are checking whether log(g) values written to the hdf5 data matches log(2)*x*y when summed under log-sum-exponentiation and if the amount of key value pairs (E->log(g)) matches the energy spectrum"""
if __name__ == "__main__":
    # check test call
    if len(sys.argv) != 2: #second param is root of execution
        print("Wrong command line parameters")
        sys.exit(-1)

    # check binaries availability
    prerun_binary_path = sys.argv[1] + "/build/src/prerun"
    mainrun_binary_path = sys.argv[1] + "/build/src/mainrun"

    if not os.path.isfile(prerun_binary_path) or not os.access(prerun_binary_path, os.X_OK) or not os.path.isfile(mainrun_binary_path) or not os.access(mainrun_binary_path, os.X_OK):
        print("Binary does not exist/can't be executed'")
        sys.exit(-1)

    # run parameter
    dimX = 5
    dimY = 5
    prob = 0.2
    num_iterations_prerun = 10000
    num_iteration_mainrun = 1000
    num_intervals_mainrun = 10
    walker_per_interval = 4
    alpha = 0.8
    beta = 0.001
    overlap = 0.25
    num_intervals_prerun = 36
    num_walker = 128
    num_interactions = 10
    replica_exchange_offsets = 20
    logical_error_type_lst = ["i", "v", "h", "c"]
    logical_error_type = random.choice(logical_error_type_lst)
    seed_prerun = int(time.time())
    seed_mainrun = seed_prerun
    task_id = 99
    time_limit = 3600

    # Test the periodic boundary execution
    boundary = 0
    checks_passed_periodic, prerun_energies, mainrun_energies, mainrun_log_g_values, prerun_arguments, mainrun_arguments = test(dimX, dimY, prob, num_iterations_prerun,
                                                                                                                                         num_iteration_mainrun, num_intervals_mainrun,
                                                                                                                                         walker_per_interval, alpha, beta, overlap,
                                                                                                                                         num_intervals_prerun, num_walker, num_interactions,
                                                                                                                                         replica_exchange_offsets, logical_error_type,
                                                                                                                                         seed_prerun, seed_mainrun, task_id, time_limit,
                                                                                                                                         boundary)

    shutil.rmtree("init")
    shutil.rmtree("results")

    if not checks_passed_periodic[0]:
        print(f"Energy sets do not match for periodic boundary!\nExpected: {prerun_energies}\nFound: {mainrun_energies}\nPrerun parameter: {prerun_arguments}\nMainrun parameter: {mainrun_arguments}")
    if not checks_passed_periodic[1]:
        print(f"Log-sum-exp mismatch for periodic boundary! Expected: {np.log(2) * dimX * dimY}, Found: {log_sum_exp(mainrun_log_g_values)}\nPrerun parameter: {prerun_arguments}\nMainrun parameter: {mainrun_arguments}")


    #Test the open boundary execution
    boundary = 1

    checks_passed_open, prerun_energies, mainrun_energies, mainrun_log_g_values, prerun_arguments, mainrun_arguments = test(dimX, dimY, prob, num_iterations_prerun,
                                                                                                                                         num_iteration_mainrun, num_intervals_mainrun,
                                                                                                                                         walker_per_interval, alpha, beta, overlap,
                                                                                                                                         num_intervals_prerun, num_walker, num_interactions,
                                                                                                                                         replica_exchange_offsets, logical_error_type,
                                                                                                                                         seed_prerun, seed_mainrun, task_id, time_limit,
                                                                                                                                         boundary)

    shutil.rmtree("init")
    shutil.rmtree("results")

    if not checks_passed_open[0]:
        print(f"Energy sets do not match for open boundary!\nExpected: {prerun_energies}\nFound: {mainrun_energies}\nPrerun parameter: {prerun_arguments}\nMainrun parameter: {mainrun_arguments}")
    if not checks_passed_open[1]:
        print(f"Log-sum-exp mismatch for open boundary! Expected: {np.log(2) * dimX * dimY}, Found: {log_sum_exp(mainrun_log_g_values)}\nPrerun parameter: {prerun_arguments}\nMainrun parameter: {mainrun_arguments}")



    if (all(checks_passed_periodic) and all(checks_passed_open)):
        sys.exit(0)
    else:
        sys.exit(-1)
