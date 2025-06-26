# WL Decoder
This repository contains an implementation of the Replica Exchange Wang-Landau algorithm used in [1]. The
Wang-Landau algorithm is a Monte Carlo simulation method designed to efficiently estimate the
density of state of a classical spin system. Our code is applicable for rectangular spin
configurations with nearest neighbor interactions and open or periodic boundary conditions. The
replica exchange improves sampling efficiency by allowing multiple replicas at different energy
ranges to exchange configurations, helping to overcome energy barriers and accelerate convergence.

The code is structured into a two-stage process: a pre-run for energy spectrum determination and a
main run for estimating the density of states.

## Dependencies

Libraries:

* CUDA, ideally 12.6, other versions may or may not work
* Boost, reasonably new version, 1.71 and 1.83 tested
* libhdf5/libhdf5-cpp

Build:

* cmake, 3.20 or newer
* gcc 13 is tested, clang 17 should work. Newer versions of both as well. msvc should work as well,
    for recent versions
* make or ninja should both be fine as generators

Code is compiled for C++20 and CUDA architectures 80 (A100) and 86 (All Ampere cards, e.g. RTX A6000).

## Build

```
mkdir build
cd build/
cmake ..
make -j 8
```
The number after `-j` indicates the number of parallel compilations `make` starts.

Currently two programs are created:
* `build/src/prerun` is the pre-run, build from the new code
* `build/src/mainrun` is the main run, build from the new code (WIP).

## Usage
The Wang-Landau simulation is performed in two stages: the pre-run and the main run.

### Pre-run
The pre-run determines the energy spectrum and initializes quenched disorder samples.
An example call looks like:

```bash
./build/src/prerun -x 6 -y 6 -p 0.04 -n 1000 -w 128 -i 32 -e "v" -r 100 -d 1 --boundary 0 --seed 12345
```

#### parameters:
- `-x`, `-y`: Lattice dimensions (x and y)
- `-p`: Error probability
- `-n`: Number of iterations
- `-w`: Number of independent walkers
- `-i`: Number of non overlapping energy intervals
- `-e`: Insertion of logical error:
    - "i": No logical error insertion
    - "v": Vertical error chain insertion
    - "h": Horizontal error chain insertion
    - "c": Combination of horizontal and vertical error insertion
- `-r`: Number of disorder samples
- `-d`: Task ID for output organization
- `--boundary`: Boundary type:
    - 0: Periodic
    - 1: Open
- `--seed`: Random seed

### Main run
The main run performs the Wang-Landau algorithm with replica exchange:

```bash
./build/src/mainrun -x 6 -y 6 -n 1000 -p 0.04 -a 0.8 -b 0.001 -i 10 -w 3 -o 0.25 -e "v" -r 100 -c 20 -d 1 --boundary 0 --seed 12345 --time_limit 3600
```

#### Additional parameters for main run:
- `-a`: Flatness parameter
- `-b`: Update factor
- `-o`: Overlap between neighboring energy intervals
- `-c`: Identifier after how many Wang-Landau iterations a replica exchange step will be executed
- `--time_limit`: Maximum runtime in seconds

Note: The pre-run must be completed before running the main run for the same seed and number of disorder samples, as the main run uses data generated during the pre-run.

## Testing

Currently three types of tests are available:

* Unit tests: Test individual functions
* Integration tests: Run an entire program end-to-end and validate its output
* Sanitizer tests: Run Nvidia's compute sanitizer on the CUDA related unit tests

By default, only the first two are enabled, to execute them, after the code was built, run

```
ctest
```
In the `build/` directory. To run either unit or integration tests, run `ctest -L unit` or
`ctest -L integration`.

### Compute sanitizer tests

Nvidia's compute sanitizer checks for memory errors, leaks, access to uninitialized device memory,
thread hazards, race conditions and synchronization problems. Since this takes a while, it's not run
per default on the CUDA related unit tests. To enable them, from `build/` run:

```
cmake -DSAN_CU_TESTS=ON ..
```
Afterwards they will be included in the tests run via `ctest`. To run them individually, run
`ctest -L sanitize`. To deactivate them again:

```
cmake -DSAN_CU_TESTS=OFF ..
```

## Output Data

The Wang-Landau simulation produces output data in HDF5 format, stored in two primary files:

1. **Prerun Output**: Contains energy spectrum data, lattice configurations with corresponding energy, and disorder samples
2. **Mainrun Output**: Contains the logarithm of the density of states for the various disorder samples

### Directory Structure

The output is organized in a hierarchical directory structure based on the simulation parameters:

```
├── init/                                # Prerun data
│   └── task_id_{task_id}/
│       └── seed_{seed}/
│           └── {boundary_type}/
│               └── prob_{probability}/
│                   └── X_{x}_Y_{y}/
│                       └── error_class_{error_type}/
│                           └── prerun_results.h5    # HDF5 file with prerun results
│
└── results/                             # Mainrun data
    └── task_id_{task_id}/
        └── seed_{seed}/
            └── {boundary_type}/
                └── prob_{probability}/
                    └── X_{x}_Y_{y}/
                        └── error_class_{error_type}/
                            └── mainrun_results.h5   # HDF5 file with mainrun results
```

Where:
- `{task_id}`: Task identifier for organizing multiple runs
- `{seed}`: Random seed
- `{boundary_type}`: Either "periodic" or "open" depending on boundary conditions
- `{probability}`: Error probability value used in the simulation
- `{x}`, `{y}`: Lattice dimensions
- `{error_type}`: Type of logical error inserted (i, v, h, or c)

### Prerun Output Structure

The prerun generates an HDF5 file with the following datasets:

- **Histogram**: Binary information about whether specific energy levels are reachable
- **Interaction**: Stores the interaction configurations for all disorder samples
- **Lattice**: Contains lattice configurations and their corresponding energies

This data serves as input to the main run and enables reproducible simulations across multiple runs.

### Mainrun Output Structure

The mainrun results are stored within the HDF5 file (`mainrun_results.h5`) with a hierarchical structure:

```
/disorder_id/seed/timestamp/log_g
```

Where:
- `disorder_id`: Identifier for the disorder sample
- `seed`: Random seed used for the simulation
- `timestamp`: Execution time of the simulation
- `log_g`: Dataset containing two columns:
  - `Key`: Energy values
  - `Value`: Logarithm of the density of states for each energy

## License

Licensed under the BSD 3-clause license, see `LICENSE` for details.

## Acknowledgments

This work was funded by the German Ministry of Economic Affairs and Climate Action (BMWK) and the
German Aerospace Center (DLR) in project QuDA-KI under grant no. 50RA2206 and through a
Leverhulme-Peierls Fellowship at the University of Oxford, funded by grant no. LIP-2020-014.

## References

[1] Wichette, L., Hohenfeld, H., Mounzer, E., & Grans-Samuelsson, L. (2025). *A partition function
framework for estimating logical error curves in stabilizer codes*.  arXiv preprint
[arXiv:2505.15758](https://arxiv.org/abs/2505.15758).
