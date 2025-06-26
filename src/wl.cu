#include <cstddef>
#include <cstdint>
#include <cstdio>

#include <cassert>
#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

#include "boundary.hpp"
#include "cuda_utils.cuh"
#include "wl.cuh"

// NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic)

namespace {

/**
 * Struct used to return the position of the flipped spin plus the new energy.
 *
 */
struct RBIM
{
  int new_energy;
  int i;
  int j;
};

/**
 * Atomic CAS for single-precision floating point
 */
__device__ float atomicCAS_f32(float* p, float cmp, float val)
{
  return __int_as_float(atomicCAS((int*)p, __float_as_int(cmp), __float_as_int(val)));
}

/**
 * @brief
 *
 * @param d_lattice Device array containing spin configurations
 * @param d_interactions Device array containing interaction configurations
 * @param d_energy Device array containing lattice energies
 * @param d_offset_lattice_per_walker Device array containing offsets for each lattice
 * @param d_seed_offset Device array containing seed offsets for each walker
 * @param rand_state Curand random state for random number generator
 * @param tid Thread ID
 * @param dimX X dimension of spin lattice
 * @param dimY Y dimension of spin lattice
 * @param interaction_offset Offset to index correct interaction lattice
 */
__device__ RBIM periodic_boundary_random_bond_ising(const int8_t* d_lattice,
                                                    const int8_t* d_interactions,
                                                    const int* d_energy,
                                                    const int* d_offset_lattice_per_walker,
                                                    unsigned long* d_seed_offset,
                                                    curandStatePhilox4_32_10_t* rand_state,
                                                    unsigned int tid,
                                                    int dimX,
                                                    int dimY,
                                                    unsigned int interaction_offset)
{
  double randval = curand_uniform(rand_state);
  randval *= (dimX * dimY - 1 + 0.999999);
  const int spin_index = (int)trunc(randval);

  d_seed_offset[tid] += 1;

  const int i = spin_index / dimY;
  const int j = spin_index % dimY;

  const int ipp = (i + 1 < dimX) ? i + 1 : 0;
  const int inn = (i - 1 >= 0) ? i - 1 : dimX - 1;
  const int jpp = (j + 1 < dimY) ? j + 1 : 0;
  const int jnn = (j - 1 >= 0) ? j - 1 : dimY - 1;

  const int up_contribution = d_lattice[d_offset_lattice_per_walker[tid] + inn * dimY + j] *
                              d_interactions[interaction_offset + dimX * dimY + inn * dimY + j];

  const int left_contribution =
    d_lattice[d_offset_lattice_per_walker[tid] + i * dimY + jnn] * d_interactions[interaction_offset + i * dimY + jnn];

  const int down_contribution = d_lattice[d_offset_lattice_per_walker[tid] + ipp * dimY + j] *
                                d_interactions[interaction_offset + dimX * dimY + i * dimY + j];

  const int right_contribution =
    d_lattice[d_offset_lattice_per_walker[tid] + i * dimY + jpp] * d_interactions[interaction_offset + i * dimY + j];

  const int energy_diff = -2 * d_lattice[d_offset_lattice_per_walker[tid] + i * dimY + j] *
                          (up_contribution + left_contribution + down_contribution + right_contribution);

  const int d_new_energy = d_energy[tid] - energy_diff;

  return {d_new_energy, i, j};
}


/**
 * @brief Compute engery difference for a randomly flipped spin with open boundary conditions.
 *
 * @param d_lattice Device array containing spin configurations
 * @param d_interactions Device array containing interaction configurations
 * @param d_energy Device array containing lattice energies
 * @param d_offset_lattice Device array containing offsets for each lattice
 * @param d_seed_offset Device array containing seed offsets for each walker
 * @param rand_state Curand random state for random number generator
 * @param tid Thread ID
 * @param dimX X dimension of spin lattice
 * @param dimY Y dimension of spin lattice
 * @param interaction_offset Offset to index correct interaction lattice
 */
__device__ RBIM open_boundary_random_bond_ising(const int8_t* d_lattice,
                                                const int8_t* d_interactions,
                                                const int* d_energy,
                                                const int* d_offset_lattice,
                                                unsigned long* d_seed_offset,
                                                curandStatePhilox4_32_10_t* rand_state,
                                                unsigned int tid,
                                                int dimX,
                                                int dimY,
                                                unsigned int interaction_offset)
{
  double randval = curand_uniform(rand_state);
  randval *= (dimX * dimY - 1 + 0.999999);
  int spin_index = (int)trunc(randval);

  d_seed_offset[tid] += 1;

  int i = spin_index / dimY;
  int j = spin_index % dimY;

  int ipp = (i + 1 < dimX) ? i + 1 : 0;
  int inn = (i - 1 >= 0) ? i - 1 : dimX - 1;
  int jpp = (j + 1 < dimY) ? j + 1 : 0;
  int jnn = (j - 1 >= 0) ? j - 1 : dimY - 1;

  int c_up = 1 - inn / (dimX - 1);
  int c_down = 1 - (i + 1) / dimX;
  int c_right = (j == (dimY - 1)) ? 0 : 1;
  int c_left = (j == 0) ? 0 : 1;

  int up_contribution = c_up * d_lattice[d_offset_lattice[tid] + inn * dimY + j] *
                        d_interactions[interaction_offset + dimX * dimY + inn * dimY + j];
  int left_contribution = c_left * d_lattice[d_offset_lattice[tid] + i * dimY + jnn] *
                          d_interactions[interaction_offset + i * dimY + jnn];
  int down_contribution = c_down * d_lattice[d_offset_lattice[tid] + ipp * dimY + j] *
                          d_interactions[interaction_offset + dimX * dimY + i * dimY + j];
  int right_contribution = c_right * d_lattice[d_offset_lattice[tid] + i * dimY + jpp] *
                          d_interactions[interaction_offset + i * dimY + j];

  int energy_diff = -2 * d_lattice[d_offset_lattice[tid] + i * dimY + j] *
                    (up_contribution + left_contribution + down_contribution + right_contribution);
  int d_new_energy = d_energy[tid] - energy_diff;

  return {d_new_energy, i, j};
}

/**
 * @brief Compute engery difference for a randomly flipped spin
 *
 * @param boundary The boundary conditions, 0 -> periodic, 1 -> open
 * @param d_lattice Device array containing spin configurations
 * @param d_interactions Device array containing interaction configurations
 * @param d_energy Device array containing lattice energies
 * @param d_offset_lattice Device array containing offsets for each lattice
 * @param d_seed_offset Device array containing seed offsets for each walker
 * @param rand_state Curand random state for random number generator
 * @param tid Thread ID
 * @param dimX X dimension of spin lattice
 * @param dimY Y dimension of spin lattice
 * @param interaction_offset Offset to index correct interaction lattice
 */
__device__ RBIM compute_new_energy(wl::boundary boundary,
                                   const int8_t* d_lattice,
                                   const int8_t* d_interactions,
                                   const int* d_energy,
                                   const int* d_offset_lattice,
                                   unsigned long* d_seed_offset,
                                   curandStatePhilox4_32_10_t* rand_state,
                                   unsigned int tid,
                                   int dimX,
                                   int dimY,
                                   int interaction_offset)
{
  if (boundary == wl::boundary::open) {
    return open_boundary_random_bond_ising(d_lattice,
                                           d_interactions,
                                           d_energy,
                                           d_offset_lattice,
                                           d_seed_offset,
                                           rand_state,
                                           tid,
                                           dimX,
                                           dimY,
                                           interaction_offset);
  } else {
    return periodic_boundary_random_bond_ising(d_lattice,
                                           d_interactions,
                                           d_energy,
                                           d_offset_lattice,
                                           d_seed_offset,
                                           rand_state,
                                           tid,
                                           dimX,
                                           dimY,
                                           interaction_offset);
  }
}


/**
 * @brief Store lattices within a certain energy interval in order to write these out to disc.
 *
 * @param d_flag_found_interval Device array indicating whether a lattice was already found for this interval.
 * @param d_found_interval_lattice Device array to write in the found lattice
 * @param d_lattice Device array containing the spin lattices
 * @param d_energy Device array containing the energies per lattice
 * @param d_offset_lattice_per_walker Device array containing the lattice offsets
 * @param E_min Minimum Energy to obtain
 * @param dimX X dimension of spin lattice
 * @param dimY Y dimension of spin lattice
 * @param tid Thread ID
 * @param len_interval Length of a single interval. Note every interval except last one has same length.
 * @param num_intervals Number of intervals to find.
 * @param interaction_id Interaction id to which the tid belongs.
 */
__device__ void store_lattice(int* d_flag_found_interval,
                              signed char* d_found_interval_lattice,
                              const signed char* d_lattice,
                              const int* d_energy,
                              const int* d_offset_lattice_per_walker,
                              int E_min,
                              int dimX,
                              int dimY,
                              unsigned int tid,
                              int len_interval,
                              unsigned int num_intervals,
                              unsigned int interaction_id)
{
  unsigned int interval_index = ((d_energy[tid] - E_min) / len_interval < num_intervals)
                                  ? (d_energy[tid] - E_min) / len_interval
                                  : num_intervals - 1;

  if (atomicCAS(&d_flag_found_interval[interaction_id * num_intervals + interval_index], 0, 1) != 0) {
    return;
  }

  const unsigned int interaction_offset = interaction_id * num_intervals * dimX * dimY;
  const unsigned int interval_offset = interval_index * dimX * dimY;

  for (int i = 0; i < dimX; i++) {
    for (int j = 0; j < dimY; j++) {
      d_found_interval_lattice[interaction_offset + interval_offset + i * dimY + j] =
        d_lattice[d_offset_lattice_per_walker[tid] + i * dimY + j];
    }
  }
}

}

__global__ void wl::calc_energy_periodic_boundary(int* d_energy,
                                                  const signed char* d_lattice,
                                                  const signed char* d_interactions,
                                                  const int* d_offset_lattice_per_walker,
                                                  int dimX,
                                                  int dimY,
                                                  std::size_t num_lattices,
                                                  int walker_per_interactions)
{
  const unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;

  if (tid >= num_lattices) {
    return;
  }

  const int interaction_id = static_cast<int>(tid) / walker_per_interactions;

  int energy = 0;

  for (int l = 0; l < dimX * dimY; l++) {
    const int i = l / dimY;
    const int j = l % dimY;

    const int inn = (i - 1 >= 0) ? i - 1 : dimX - 1;
    const int jnn = (j - 1 >= 0) ? j - 1 : dimY - 1;

    const int up_spin_index = d_offset_lattice_per_walker[tid] + inn * dimY + j;
    const int up_interaction_index = interaction_id * 2 * dimX * dimY + dimX * dimY + inn * dimY + j;

    const int left_spin_index = d_offset_lattice_per_walker[tid] + i * dimY + jnn;
    const int left_interaction_index = interaction_id * dimX * dimY * 2 + i * dimY + jnn;

    const int up_contribution = d_lattice[up_spin_index] * d_interactions[up_interaction_index];
    const int left_contribution = d_lattice[left_spin_index] * d_interactions[left_interaction_index];

    energy += d_lattice[d_offset_lattice_per_walker[tid] + i * dimY + j] * (up_contribution + left_contribution);
  }

  d_energy[tid] = -energy;
}

__global__ void wl::calc_energy_open_boundary(int* d_energy,
                                              const signed char* d_lattice,
                                              const signed char* d_interactions,
                                              const int* d_offset_lattice,
                                              const int dimX,
                                              const int dimY,
                                              std::size_t num_lattices,
                                              int walker_per_interactions)
{
  const unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;

  if (tid >= num_lattices) {
    return;
  }

  const int interaction_id = static_cast<int>(tid) / walker_per_interactions;

  int energy = 0;
  int offset = d_offset_lattice[tid];

  for (int i = 0; i < dimX; i++) {
    for (int j = 0; j < dimY; j++) {
      int8_t spin = d_lattice[offset + i * dimY + j];
      int8_t up_spin = (i > 0) ? d_lattice[offset + (i - 1) * dimY + j] : 0;
      int8_t left_spin = (j > 0) ? d_lattice[offset + i * dimY + (j - 1)] : 0;

      // to avoid accessing interactions out of range for boundary terms the indices are arbitrarily set to 0
      int inn = (i > 0) ? dimX * dimY + (i - 1) * dimY + j : 0;
      int jnn = (j > 0) ? i * dimY + (j - 1) : 0;

      int8_t up_interaction = d_interactions[interaction_id * dimX * dimY * 2 + inn];
      int8_t left_interaction = d_interactions[interaction_id * dimX * dimY * 2 + jnn];

      energy += spin * (up_spin * up_interaction + left_spin * left_interaction);
    }
  }

  d_energy[tid] = -energy;
}

void wl::calc_energy(unsigned int blocks,
                     unsigned int threads,
                     wl::boundary boundary,
                     int* d_energy,
                     const signed char* d_lattice,
                     const signed char* d_interactions,
                     const int* d_offset_lattice,
                     int dimX,
                     int dimY,
                     std::size_t num_lattices,
                     int walker_per_interactions)
{
  if (boundary == wl::boundary::open) {
    calc_energy_open_boundary<<<blocks, threads>>>(
        d_energy, d_lattice, d_interactions,  d_offset_lattice, dimX, dimY, num_lattices, walker_per_interactions);
  } else {
    calc_energy_periodic_boundary<<<blocks, threads>>>(
        d_energy, d_lattice, d_interactions, d_offset_lattice, dimX, dimY, num_lattices, walker_per_interactions);

  }
}

__global__ void wl::init_interactions(int8_t* interactions,
                                      size_t lattice_size,
                                      int num_interactions,
                                      int seed,
                                      float prob)
{
  std::size_t tid = static_cast<size_t>(blockDim.x) * blockIdx.x + threadIdx.x;
  auto interaction_id = static_cast<size_t>(tid / (lattice_size * 2));

  if (tid >= lattice_size * 2 * num_interactions) {
    return;
  }

  auto lin_interaction_idx = static_cast<size_t>(tid % (lattice_size * 2));

  curandStatePhilox4_32_10_t state;
  curand_init(seed + interaction_id, 0, lin_interaction_idx, &state);

  double randval = curand_uniform(&state);
  signed char val = (randval < prob) ? -1 : 1;

  interactions[tid] = val;
}

__global__ void wl::apply_x_horizontal_error(signed char* interactions,
                                             int lattice_dim_x,
                                             int lattice_dim_y,
                                             int num_interactions)
{
  std::size_t tid = static_cast<long long>(blockDim.x) * blockIdx.x + threadIdx.x;

  if (tid >= lattice_dim_y * num_interactions) {
    return;
  }

  auto disorder_sample_interaction_offset =
    static_cast<size_t>(static_cast<int>(tid / lattice_dim_y) * 2 * lattice_dim_x * lattice_dim_y);

  auto col = static_cast<size_t>(tid % lattice_dim_y);

  // flip all interactions in row = lattice_dim_x for all disorder samples - this row carries the up interactions of
  // first row
  interactions[disorder_sample_interaction_offset + lattice_dim_x * lattice_dim_y + col] *= -1;
}

__global__ void wl::apply_x_vertical_error(signed char* interactions,
                                           int lattice_dim_x,
                                           int lattice_dim_y,
                                           int num_interactions)
{
  std::size_t tid = static_cast<long long>(blockDim.x) * blockIdx.x + threadIdx.x;

  if (tid >= lattice_dim_x * num_interactions) {
    return;
  }

  auto disorder_sample_interaction_offset =
    static_cast<size_t>(static_cast<int>(tid / lattice_dim_x) * 2 * lattice_dim_x * lattice_dim_y);

  auto row = static_cast<size_t>(tid % lattice_dim_x);

  // flip all interactions in first colum in first lattice_dim_x rows for all disorder samples - this row carries the up
  // interactions of first row
  interactions[disorder_sample_interaction_offset + row * lattice_dim_y] *= -1;
}

__global__ void wl::init_lattice(signed char* d_lattice,
                                 float* d_init_probs_per_lattice,
                                 int dimX,
                                 int dimY,
                                 std::size_t num_lattices,
                                 int seed,
                                 int walker_per_interactions)
{
  const unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;

  if (tid >= dimX * dimY * num_lattices) {
    return;
  }

  const unsigned int lattice_id = tid / (dimX * dimY);
  const unsigned int interaction_id = lattice_id / walker_per_interactions;

  curandStatePhilox4_32_10_t state;
  curand_init(seed + interaction_id, 1, tid % (dimX * dimY), &state);

  atomicCAS_f32(&d_init_probs_per_lattice[lattice_id], 0.0F, curand_uniform(&state));
  __syncthreads();

  double randval = curand_uniform(&state);
  signed char val = (randval < d_init_probs_per_lattice[lattice_id]) ? -1 : 1;

  d_lattice[tid] = val;
}

__global__ void wl::init_offsets_lattice(int* offset_lattice,
                                         int lattice_dim_x,
                                         int lattice_dim_y,
                                         std::size_t num_lattices)
{
  unsigned int tid = static_cast<unsigned int>(blockDim.x) * blockIdx.x + threadIdx.x;

  if (tid >= num_lattices) {
    return;
  }

  offset_lattice[tid] = static_cast<int>(tid * lattice_dim_x * lattice_dim_y);
}

__global__ void wl::wang_landau_pre_run(int* d_energy,
                                        int* d_flag_found_interval,
                                        int8_t* d_lattice,
                                        int8_t* d_found_interval_lattice,
                                        unsigned long long* d_histogram,
                                        unsigned long* d_seed_offset,
                                        const signed char* d_interactions,
                                        const int* d_offset_lattice_per_walker,
                                        int E_min,
                                        int E_max,
                                        int num_wl_iterations,
                                        int dimX,
                                        int dimY,
                                        int seed,
                                        unsigned int num_intervals,
                                        int len_interval,
                                        std::size_t num_total_walker,
                                        int walker_per_interactions,
                                        wl::boundary boundary)
{
  const unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;

  if (tid >= num_total_walker) {
    return;
  }

  const unsigned int interaction_id = tid / walker_per_interactions;
  const unsigned int interaction_offset = interaction_id * 2 * dimX * dimY;

  const int len_hist = E_max - E_min + 1;

  curandStatePhilox4_32_10_t rand_state;
  curand_init(seed + interaction_id, 2 + (tid % walker_per_interactions), d_seed_offset[tid], &rand_state);

  for (int it = 0; it < num_wl_iterations; it++) {
    RBIM result = compute_new_energy(boundary,
                                     d_lattice,
                                     d_interactions,
                                     d_energy,
                                     d_offset_lattice_per_walker,
                                     d_seed_offset,
                                     &rand_state,
                                     tid,
                                     dimX,
                                     dimY,
                                     interaction_offset);

    const int d_new_energy = result.new_energy;

    if (d_new_energy > E_max || d_new_energy < E_min) {
      printf("Iterator %d \n", it);
      printf("Thread Id %d \n", tid);
      printf("Energy out of range %d \n", d_new_energy);
      printf("Old energy %d \n", d_energy[tid]);
      assert(0);
      return;
    }

    const unsigned int index_old = d_energy[tid] - E_min + static_cast<int>(interaction_id) * len_hist;
    const unsigned int index_new = d_new_energy - E_min + static_cast<int>(interaction_id) * len_hist;

    double prob = exp(static_cast<double>(d_histogram[index_old]) - static_cast<double>(d_histogram[index_new]));

    if (curand_uniform(&rand_state) < prob) {
      d_lattice[d_offset_lattice_per_walker[tid] + result.i * dimY + result.j] *= -1;
      d_energy[tid] = d_new_energy;

      atomicAdd(&d_histogram[index_new], 1);

      store_lattice(d_flag_found_interval,
                    d_found_interval_lattice,
                    d_lattice,
                    d_energy,
                    d_offset_lattice_per_walker,
                    E_min,
                    dimX,
                    dimY,
                    tid,
                    len_interval,
                    num_intervals,
                    interaction_id);
    } else {
      atomicAdd(&d_histogram[index_old], 1);
    }

    d_seed_offset[tid] += 1;
  }
}

__global__ void wl::check_energy_ranges(int8_t* d_flag_check_energies,
                                        const int* d_energy_per_walker,
                                        const int* d_interval_start_energies,
                                        const int* d_interval_end_energies,
                                        std::size_t num_total_walker)
{
  const unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;

  if (tid >= num_total_walker) {
    return;
  }

  if (d_energy_per_walker[tid] > d_interval_end_energies[blockIdx.x] ||
      d_energy_per_walker[tid] < d_interval_start_energies[blockIdx.x]) {
    d_flag_check_energies[tid] = 1;
    printf("tid=%d energy=%d start_interval=%d end_interval=%d  \n",
           tid,
           d_energy_per_walker[tid],
           d_interval_start_energies[blockIdx.x],
           d_interval_end_energies[blockIdx.x]);
  }
}

__global__ void wl::init_indices(int* d_replica_indices, std::size_t num_total_walker)
{
  const unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;

  if (tid >= num_total_walker) {
    return;
  }

  d_replica_indices[tid] = static_cast<int>(threadIdx.x);
}

__global__ void wl::reset_d_cond(int8_t* d_cond_interval,
                                 const double* d_factor,
                                 int total_intervals,
                                 double beta,
                                 int walker_per_interval)
{
  unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;

  if (tid >= total_intervals) {
    return;
  }

  if (d_cond_interval[tid] == 1) {
    if (d_factor[tid * walker_per_interval] > exp(beta)) {
      d_cond_interval[tid] = 0;
    }
  }
}

__global__ void wl::init_offsets_histogram(int* d_offset_histogram,
                                           const int* d_interval_start_energies,
                                           const int* d_interval_end_energies,
                                           const int* d_len_histograms,
                                           int num_intervals,
                                           int total_walker)
{
  unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;

  if (tid >= total_walker) {
    return;
  }

  unsigned int interaction_id = static_cast<int>(blockIdx.x) / num_intervals;

  // Offset of interactions before
  int offset_hist = 0;
  for (int i = 0; i < interaction_id; i++) {
    offset_hist += d_len_histograms[i];
  }

  int len_first_intervals = d_interval_end_energies[interaction_id * num_intervals] -
                            d_interval_start_energies[interaction_id * num_intervals] + 1;

  // Check if I am in the last interval, as this has different length
  if (blockIdx.x % num_intervals == (num_intervals - 1)) {
    int len_last_interval = d_interval_end_energies[(interaction_id + 1) * num_intervals - 1] -
                            d_interval_start_energies[(interaction_id + 1) * num_intervals - 1] + 1;
    int offset_first_intervals = (num_intervals - 1) * static_cast<int>(blockDim.x) * len_first_intervals;

    d_offset_histogram[tid] = offset_hist + offset_first_intervals + static_cast<int>(threadIdx.x) * len_last_interval;
  } else {
    // Bracket is equal to walker_per_interaction --> Walker_id is which walker I am on the interaction
    int walker_id = static_cast<int>(tid) % (num_intervals * static_cast<int>(blockDim.x));

    d_offset_histogram[tid] = offset_hist + static_cast<int>(walker_id) * len_first_intervals;
  }
}

/**
 * @brief Checks whether all intervals in an interaction is finished.
 *
 *
 * @param d_cond_interactions Array with flags for each interaction
 * @param num_intervals Number of intervals per interaction
 * @param num_interactions Number of interactions
 */
__global__ void check_all_intervals_finished(int* d_cond_interactions, int num_intervals, int num_interactions)
{
  unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;

  if (tid >= num_interactions) {
    return;
  }

  if (d_cond_interactions[tid] == num_intervals) {
    d_cond_interactions[tid] = -1;
  }
}

void wl::check_interactions_finished(int* d_cond_interactions,
                                     wl::device_tmp<void>& d_tmp_storage,
                                     const int8_t* d_cond_interval,
                                     const int* d_offset_intervals,
                                     int num_intervals,
                                     int num_interactions)
{
  size_t needed_storage_bytes = 0;

  // Determine the amount of temporary storage needed
  cub::DeviceSegmentedReduce::Sum(nullptr,
                                  needed_storage_bytes,
                                  d_cond_interval,
                                  d_cond_interactions,
                                  num_interactions,
                                  d_offset_intervals,
                                  d_offset_intervals + 1);

  d_tmp_storage.resize(needed_storage_bytes);

  // Perform the segmented reduction
  cub::DeviceSegmentedReduce::Sum(d_tmp_storage.data(),
                                  needed_storage_bytes,
                                  // temp_storage_bytes,
                                  d_cond_interval,
                                  d_cond_interactions,
                                  num_interactions,
                                  d_offset_intervals,
                                  d_offset_intervals + 1);
  cudaDeviceSynchronize();

  check_all_intervals_finished<<<num_interactions, 1>>>(d_cond_interactions, num_intervals, num_interactions);
  cudaDeviceSynchronize();
}

__global__ void wl::calc_average_log_g(double* d_shared_logG,
                                       const int* d_len_histograms,
                                       const double* d_log_G,
                                       const int* d_interval_start_energies,
                                       const int* d_interval_end_energies,
                                       const int8_t* d_expected_energy_spectrum,
                                       const int8_t* d_cond,
                                       const int* d_offset_histogram,
                                       const int* d_offset_energy_spectrum,
                                       const long long* d_offset_shared_logG,
                                       const int* d_cond_interaction,
                                       int num_interactions,
                                       int num_walker_per_interval,
                                       int num_intervals_per_interaction,
                                       int total_len_histogram)
{
  unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;

  if (tid >= total_len_histogram) {
    return;
  }

  // Calc interaction id
  int interaction_id = 0;
  for (int i = 0; i < num_interactions; i++) {
    if (i == num_interactions - 1) {
      interaction_id = i;
    }
    if (tid < d_offset_histogram[(i + 1) * num_intervals_per_interaction * num_walker_per_interval]) {
      interaction_id = i;
      break;
    }
  }

  if (d_cond_interaction[interaction_id] == -1) {
    return;
  }

  // Index inside histogram of the interaction_id interaction
  int tid_int = static_cast<int>(tid) -
                d_offset_histogram[interaction_id * num_intervals_per_interaction * num_walker_per_interval];

  int len_first_interval = (d_interval_end_energies[interaction_id * num_intervals_per_interaction] -
                            d_interval_start_energies[interaction_id * num_intervals_per_interaction] + 1);

  int intervalId = 0;

  if (tid_int / (len_first_interval * num_walker_per_interval) < num_intervals_per_interaction) {
    intervalId = tid_int / (len_first_interval * num_walker_per_interval);
  } else {
    intervalId = num_intervals_per_interaction - 1;
  }

  int interval_over_interaction = interaction_id * num_intervals_per_interaction + intervalId;

  if (d_cond[interval_over_interaction] == 1 && tid_int < d_len_histograms[interaction_id]) {
    int len_interval =
      d_interval_end_energies[interval_over_interaction] - d_interval_start_energies[interval_over_interaction] + 1;
    int energyId{};

    if (intervalId != 0) {
      energyId = (tid_int % (len_first_interval * num_walker_per_interval * intervalId)) % len_interval;
    } else {
      energyId = tid_int % len_interval;
    }

    if (d_expected_energy_spectrum[d_offset_energy_spectrum[interaction_id] +
                                   d_interval_start_energies[interval_over_interaction] + energyId -
                                   d_interval_start_energies[interaction_id * num_intervals_per_interaction]] == 1) {
      atomicAdd(&d_shared_logG[d_offset_shared_logG[interval_over_interaction] + energyId],
                d_log_G[tid] / num_walker_per_interval);
    }
  }
}

__device__ void wl::fisher_yates(int* d_shuffle,
                             unsigned long* d_seed_offset,
                             int seed,
                             int interaction_id,
                             int walker_per_interactions)
{
  unsigned long tid = blockDim.x * blockIdx.x + threadIdx.x;

  int offset = static_cast<int>(blockDim.x) * blockIdx.x;

  curandStatePhilox4_32_10_t rand_state;
  curand_init(seed + interaction_id, tid % walker_per_interactions, d_seed_offset[tid], &rand_state);

  for (int i = static_cast<int>(blockDim.x) - 1; i > 0; i--) {
    double randval = curand_uniform(&rand_state);
    randval *= (i + 0.999999);
    int random_index = (int)trunc(randval);
    d_seed_offset[tid] += 1;

    int temp = d_shuffle[offset + i];
    d_shuffle[offset + i] = d_shuffle[offset + random_index];
    d_shuffle[offset + random_index] = temp;
  }
}

__global__ void wl::replica_exchange(int* d_offset_lattice,
                                     int* d_energy,
                                     int* d_replica_indices,
                                     unsigned long* d_seed_offset,
                                     const int* d_interval_start_energies,
                                     const int* d_interval_end_energies,
                                     const int* d_offset_histogram,
                                     const int* d_cond_interaction,
                                     const double* d_logG,
                                     bool even,
                                     int seed,
                                     int num_intervals,
                                     int walker_per_interactions)
{
  // if last block in interaction return
  if (blockIdx.x % num_intervals == (num_intervals - 1)) {
    return;
  }

  // if even only even blocks if odd only odd blocks
  if ((even && (blockIdx.x % 2 != 0)) || (!even && (blockIdx.x % 2 == 0))) {
    return;
  }

  unsigned long tid = blockDim.x * blockIdx.x + threadIdx.x;
  const int interaction_id = static_cast<int>(tid) / walker_per_interactions;

  if (d_cond_interaction[interaction_id] == -1) {
    return;
  }

  unsigned long swap_id = static_cast<long long>(blockDim.x) * (blockIdx.x + 1);

  if (threadIdx.x == 0) {
    fisher_yates(d_replica_indices, d_seed_offset, seed, interaction_id, walker_per_interactions);
  }

  __syncthreads();

  swap_id += d_replica_indices[tid];

  // Check if swapped energies lie in range
  if (d_energy[tid] > d_interval_end_energies[blockIdx.x + 1] ||
      d_energy[tid] < d_interval_start_energies[blockIdx.x + 1]) {
    return;
  }

  if (d_energy[swap_id] > d_interval_end_energies[blockIdx.x] ||
      d_energy[swap_id] < d_interval_start_energies[blockIdx.x]) {
    return;
  }

  // Prob calculation according to https://iopscience.iop.org/article/10.1088/1742-6596/1012/1/012003/pdf eq (1)
  double g_i_E_X =
    d_logG[d_offset_histogram[tid] + static_cast<int>(d_energy[tid]) - d_interval_start_energies[blockIdx.x]];
  double g_i_E_Y =
    d_logG[d_offset_histogram[tid] + static_cast<int>(d_energy[swap_id]) - d_interval_start_energies[blockIdx.x]];
  double g_j_E_Y = d_logG[d_offset_histogram[swap_id] + static_cast<int>(d_energy[swap_id]) -
                          d_interval_start_energies[blockIdx.x + 1]];
  double g_j_E_X =
    d_logG[d_offset_histogram[swap_id] + static_cast<int>(d_energy[tid]) - d_interval_start_energies[blockIdx.x + 1]];

  double prob = min(1.0, exp(g_i_E_X - g_i_E_Y) * exp(g_j_E_Y - g_j_E_X));

  curandStatePhilox4_32_10_t rand_state;
  curand_init(seed + interaction_id, tid % walker_per_interactions, d_seed_offset[tid], &rand_state);

  // Swap energies
  if (curand_uniform(&rand_state) < prob) {
    int temp_off = d_offset_lattice[tid];
    int temp_energy = d_energy[tid];

    d_offset_lattice[tid] = d_offset_lattice[swap_id];
    d_energy[tid] = d_energy[swap_id];

    d_offset_lattice[swap_id] = temp_off;
    d_energy[swap_id] = temp_energy;
  }

  d_seed_offset[tid] += 1;
}

__global__ void wl::check_histogram(unsigned long long* d_histogram,
                                    double* d_factor,
                                    int8_t* d_cond_interval,
                                    const int* d_offset_histogram,
                                    const int* d_interval_end_energies,
                                    const int* d_interval_start_energies,
                                    const int8_t* d_expected_energy_spectrum,
                                    const int* d_offset_energy_spectrum,
                                    const int* d_cond_interaction,
                                    double alpha,
                                    int num_walker_total,
                                    int walker_per_interactions,
                                    int num_intervals)
{
  unsigned long tid = static_cast<long long>(blockDim.x) * blockIdx.x + threadIdx.x;

  int interaction_id = static_cast<int>(tid) / walker_per_interactions;
  int blockId = static_cast<int>(blockIdx.x);

  if ((tid >= num_walker_total) || (d_cond_interval[blockId] == 1) || (d_cond_interaction[interaction_id] == -1)) {
    return;
  }

  __shared__ int walkers_finished;

  if (threadIdx.x == 0) {
    walkers_finished = 0;
  }

  __syncthreads();

  unsigned long min = INT_MAX;
  double average = 0;
  int len_reduced_energy_spectrum = 0;
  int len_current_interval = (d_interval_end_energies[blockId] - d_interval_start_energies[blockId] + 1);

  // Here is average and min calculation over all bins in histogram which correspond to values in expected energy
  // spectrum
  for (int i = 0; i < len_current_interval; i++) {
    if (d_expected_energy_spectrum[d_offset_energy_spectrum[interaction_id] + d_interval_start_energies[blockId] + i -
                                   d_interval_start_energies[interaction_id * num_intervals]] == 1) {
      if (d_histogram[d_offset_histogram[tid] + i] < min) {
        min = d_histogram[d_offset_histogram[tid] + i];
      }

      average += static_cast<double>(d_histogram[d_offset_histogram[tid] + i]);
      len_reduced_energy_spectrum += 1;
    }
  }

  __syncthreads();

  if (len_reduced_energy_spectrum > 0) {
    average = average / len_reduced_energy_spectrum;
    // printf("Walker %d in interval %d with min %lld average %.6f alpha %.6f alpha*average %.2f and factor %.10f and
    // d_cond_interval %d and end %d and start %d\n", threadIdx.x, blockIdx.x, min, average, alpha, alpha * average,
    // d_factor[tid], d_cond_interval[blockId], d_interval_end_energies[blockId], d_interval_start_energies[blockId]);
    if (static_cast<double>(min) >= alpha * average) {
      atomicAdd(&walkers_finished, 1);
    }
  } else {
    printf("Error histogram has no sufficient length to check for flatness on walker %lld. \n", tid);
  }

  __syncthreads();

  if (walkers_finished == blockDim.x) {
    d_cond_interval[blockId] = 1;

    for (int i = 0; i < len_current_interval; i++) {
      d_histogram[d_offset_histogram[tid] + i] = 0;
    }

    d_factor[tid] = sqrt(d_factor[tid]);
  }
}

__global__ void wl::redistribute_g_values(double* d_log_G,
                                          const double* d_shared_logG,
                                          const int* d_len_histograms,
                                          const int* d_interval_start_energies,
                                          const int* d_interval_end_energies,
                                          const int* d_cond_interaction,
                                          const int* d_offset_histogram,
                                          const int8_t* d_cond_interval,
                                          const long long* d_offset_shared_logG,
                                          int num_intervals_per_interaction,
                                          int num_walker_per_interval,
                                          int num_interactions,
                                          int total_len_histogram)
{
  unsigned long tid = blockDim.x * blockIdx.x + threadIdx.x;

  if (tid >= total_len_histogram) {
    return;
  }

  // Calc interaction_id
  int interaction_id = 0;
  for (int i = 0; i < num_interactions; i++) {
    if (i == num_interactions - 1) {
      interaction_id = i;
    }
    if (tid < d_offset_histogram[(i + 1) * num_intervals_per_interaction * num_walker_per_interval]) {
      interaction_id = i;
      break;
    }
  }

  // Check if interaction is already finished and return if true
  if (d_cond_interaction[interaction_id] == -1) {
    return;
  }

  // thread id in interaction
  int tid_int = static_cast<int>(tid) -
                d_offset_histogram[interaction_id * num_intervals_per_interaction * num_walker_per_interval];

  int len_first_interval = (d_interval_end_energies[interaction_id * num_intervals_per_interaction] -
                            d_interval_start_energies[interaction_id * num_intervals_per_interaction] + 1);
  int intervalId = (tid_int / (len_first_interval * num_walker_per_interval) < num_intervals_per_interaction)
                     ? tid_int / (len_first_interval * num_walker_per_interval)
                     : num_intervals_per_interaction - 1;
  int interval_over_interaction = interaction_id * num_intervals_per_interaction + intervalId;

  // Check if in right range
  if (tid_int < d_len_histograms[interaction_id] && d_cond_interval[interval_over_interaction] == 1) {
    int len_interval =
      d_interval_end_energies[interval_over_interaction] - d_interval_start_energies[interval_over_interaction] + 1;
    int energyId = 0;

    if (intervalId != 0) {
      energyId = (tid_int % (len_first_interval * num_walker_per_interval * intervalId)) % len_interval;
    } else {
      energyId = tid_int % len_interval;
    }

    d_log_G[tid] = d_shared_logG[d_offset_shared_logG[interval_over_interaction] + energyId];
  }
}

__global__ void wl::wang_landau(signed char* d_lattice,
                                signed char* d_interactions,
                                unsigned long long* d_H,
                                double* d_logG,
                                double* factor,
                                int* d_energy,
                                int* d_newEnergies,
                                unsigned long* d_offset_iter,
                                int* foundFlag,
                                const int* d_start,
                                const int* d_end,
                                const int* d_offset_histogram,
                                const int* d_offset_lattice,
                                const int8_t* d_cond,
                                const int8_t* d_expected_energy_spectrum,
                                const int* d_offset_energy_spectrum,
                                const int* d_cond_interaction,
                                int dimX,
                                int dimY,
                                int num_iterations,
                                int num_lattices,
                                int num_intervals,
                                int walker_per_interactions,
                                int seed,
                                wl::boundary boundary)
{
  unsigned int tid = static_cast<long long>(blockDim.x) * blockIdx.x + threadIdx.x;

  if (tid >= num_lattices) {
    return;
  }

  const unsigned int blockId = blockIdx.x;
  const unsigned int interaction_id = tid / walker_per_interactions;
  const unsigned int interaction_offset = interaction_id * 2 * dimX * dimY;

  if (d_cond_interaction[interaction_id] == -1) {
    return;
  }

  curandStatePhilox4_32_10_t random_state;
  curand_init(seed + interaction_id, tid % walker_per_interactions, d_offset_iter[tid], &random_state);

  if (d_cond[blockId] == 0) {
    for (int it = 0; it < num_iterations; it++) {
      RBIM result = compute_new_energy(boundary,
                                       d_lattice,
                                       d_interactions,
                                       d_energy,
                                       d_offset_lattice,
                                       d_offset_iter,
                                       &random_state,
                                       tid,
                                       dimX,
                                       dimY,
                                       interaction_offset);

      // If no new energy is found, set it to 0, else to tid + 1
      if (d_expected_energy_spectrum[d_offset_energy_spectrum[interaction_id] + result.new_energy -
                                     d_start[interaction_id * num_intervals]] == 1) {
        foundFlag[tid] = 0;
      } else {
        foundFlag[tid] = static_cast<int>(tid) + 1;
      }

      if (foundFlag[tid] != 0) {
        printf("new_energy %d index in spectrum %d \n",
               result.new_energy,
               result.new_energy - d_start[interaction_id * num_intervals]);
        d_newEnergies[tid] = result.new_energy;
        return;
      }

      int index_old = d_offset_histogram[tid] + d_energy[tid] - d_start[blockId];

      if (result.new_energy > d_end[blockId] || result.new_energy < d_start[blockId]) {
        d_H[index_old] += 1;
        d_logG[index_old] += log(factor[tid]);
      } else {
        int index_new = d_offset_histogram[tid] + result.new_energy - d_start[blockId];
        double prob = min(1.0, exp(d_logG[index_old] - d_logG[index_new]));
        double randval = curand_uniform(&random_state);

        if (randval < prob) {
          d_lattice[d_offset_lattice[tid] + result.i * dimY + result.j] *= -1;
          d_H[index_new] += 1;
          d_logG[index_new] += log(factor[tid]);
          d_energy[tid] = result.new_energy;
        }

        else {
          d_H[index_old] += 1;
          d_logG[index_old] += log(factor[tid]);
        }

        d_offset_iter[tid] += 1;
      }
    }
  } else {
    for (int it = 0; it < num_iterations; it++) {
      RBIM result = compute_new_energy(boundary,
                                       d_lattice,
                                       d_interactions,
                                       d_energy,
                                       d_offset_lattice,
                                       d_offset_iter,
                                       &random_state,
                                       tid,
                                       dimX,
                                       dimY,
                                       interaction_offset);

      // If no new energy is found, set it to 0, else to tid + 1
      if (d_expected_energy_spectrum[d_offset_energy_spectrum[interaction_id] + result.new_energy -
                                     d_start[interaction_id * num_intervals]] == 1) {
        foundFlag[tid] = 0;
      } else {
        foundFlag[tid] = static_cast<int>(tid) + 1;
      }

      if (foundFlag[tid] != 0) {
        printf("new_energy %d index in spectrum %d \n",
               result.new_energy,
               result.new_energy - d_start[interaction_id * num_intervals]);
        d_newEnergies[tid] = result.new_energy;
        return;
      }

      if (result.new_energy <= d_end[blockId] || result.new_energy >= d_start[blockId]) {
        int index_old = d_offset_histogram[tid] + d_energy[tid] - d_start[blockId];
        int index_new = d_offset_histogram[tid] + result.new_energy - d_start[blockId];

        double prob = min(1.0, exp(d_logG[index_old] - d_logG[index_new]));

        if (curand_uniform(&random_state) < prob) {
          d_lattice[d_offset_lattice[tid] + result.i * dimY + result.j] *= -1;
          d_energy[tid] = result.new_energy;
        }
        d_offset_iter[tid] += 1;
      }
    }
  }
}

// NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic)
