#include <boost/test/tools/old/interface.hpp>
#define BOOST_TEST_MODULE wl_tests

#include <boost/test/unit_test.hpp>
#include <cuda_runtime.h>
#include <vector>
#include <random>
#include <curand.h>
#include <curand_kernel.h>

#include "../src/wl.cuh"
#include "../src/cuda_utils.cuh"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            BOOST_FAIL("CUDA Error: " << cudaGetErrorString(err)); \
        } \
    } while (0)

struct RBIM
{
  int new_energy;
  int i;
  int j;
};

void initialize_randomly(std::vector<std::int8_t>& data) {
    std::random_device rd;
    std::mt19937 generator(rd());
    std::uniform_int_distribution<int> distribution(0, 1);

    for (auto& value : data) {
        value = distribution(generator) == 0 ? -1 : 1;
    }
}

__global__ void get_new_config_energy_and_flipped_spin_index_periodic(
                                RBIM* d_new_config,
                                signed char* d_lattice,
                                signed char* d_interactions,
                                int* d_energy,
                                unsigned long* d_offset_iter,
                                const int* d_offset_lattice,
                                int dimX,
                                int dimY,
                                int num_lattices,
                                int walker_per_interactions,
                                int seed)
{
    unsigned int tid = static_cast<long long>(blockDim.x) * blockIdx.x + threadIdx.x;

    if (tid >= num_lattices) {
        return;
    }

    const unsigned int interaction_id = tid / walker_per_interactions;
    const unsigned int interaction_offset = interaction_id * 2 * dimX * dimY;


    curandStatePhilox4_32_10_t random_state;
    curand_init(seed + interaction_id, tid % walker_per_interactions, d_offset_iter[tid], &random_state);

    double randval = curand_uniform(&random_state);
    randval *= (dimX * dimY - 1 + 0.999999);
    const int spin_index = (int)trunc(randval);

    d_offset_iter[tid] += 1;

    const int i = spin_index / dimY;
    const int j = spin_index % dimY;

    const int ipp = (i + 1 < dimX) ? i + 1 : 0;
    const int inn = (i - 1 >= 0) ? i - 1 : dimX - 1;
    const int jpp = (j + 1 < dimY) ? j + 1 : 0;
    const int jnn = (j - 1 >= 0) ? j - 1 : dimY - 1;

    const int up_contribution = d_lattice[d_offset_lattice[tid] + inn * dimY + j] *
                                d_interactions[interaction_offset + dimX * dimY + inn * dimY + j];

    const int left_contribution =
        d_lattice[d_offset_lattice[tid] + i * dimY + jnn] * d_interactions[interaction_offset + i * dimY + jnn];

    const int down_contribution = d_lattice[d_offset_lattice[tid] + ipp * dimY + j] *
                                    d_interactions[interaction_offset + dimX * dimY + i * dimY + j];

    const int right_contribution =
        d_lattice[d_offset_lattice[tid] + i * dimY + jpp] * d_interactions[interaction_offset + i * dimY + j];

    const int energy_diff = -2 * d_lattice[d_offset_lattice[tid] + i * dimY + j] *
                            (up_contribution + left_contribution + down_contribution + right_contribution);

    const int d_new_energy = d_energy[tid] - energy_diff;

    d_new_config[tid] = {d_new_energy, i, j};
}

__global__ void get_new_config_energy_and_flipped_spin_index_open(
                                RBIM* d_new_config,
                                signed char* d_lattice,
                                signed char* d_interactions,
                                int* d_energy,
                                unsigned long* d_offset_iter,
                                const int* d_offset_lattice,
                                int dimX,
                                int dimY,
                                int num_lattices,
                                int walker_per_interactions,
                                int seed)
{
    unsigned int tid = static_cast<long long>(blockDim.x) * blockIdx.x + threadIdx.x;

    if (tid >= num_lattices) {
        return;
    }

    const unsigned int interaction_id = tid / walker_per_interactions;
    const unsigned int interaction_offset = interaction_id * 2 * dimX * dimY;


    curandStatePhilox4_32_10_t random_state;
    curand_init(seed + interaction_id, tid % walker_per_interactions, d_offset_iter[tid], &random_state);

    double randval = curand_uniform(&random_state);
    randval *= (dimX * dimY - 1 + 0.999999);
    const int spin_index = (int)trunc(randval);

    d_offset_iter[tid] += 1;

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

    d_new_config[tid] = {d_new_energy, i, j};
}

BOOST_AUTO_TEST_SUITE(wang_landau)

BOOST_AUTO_TEST_CASE(test_wang_landau_periodic_success)
{
    const int num_interactions = 10;
    const int num_intervals_per_interaction = 10;
    const int num_walker_per_interval = 5;
    const int total_walker = num_interactions*num_intervals_per_interaction*num_walker_per_interval;
    const int total_intervals = num_intervals_per_interaction * num_interactions;
    const int seed = 42;
    const int dimX = 4;
    const int dimY = 4;
    const int num_iterations = 1;
    const int total_len_histogram = total_walker * 4*dimX*dimY + 1;


    // Device arrays
    wl::device_ptr<unsigned long> d_offset_iter(total_walker, 0);
    wl::device_ptr<int> d_energy(total_walker, 0);
    wl::device_ptr<RBIM> d_new_config(total_walker);
    wl::device_ptr<int> d_cond_interactions(num_interactions, 0);
    wl::device_ptr<std::int8_t> d_cond(total_intervals, 0);
    wl::device_ptr<int> d_newEnergies(total_walker);
    wl::device_ptr<int> d_foundNewEnergyFlag(total_walker);
    wl::device_ptr<unsigned long long> d_H(total_len_histogram, 0);
    wl::device_ptr<double> d_logG(total_len_histogram, 0);

    // Copy data from host to device
    std::vector<std::int8_t> h_lattice(total_walker * dimX * dimY);
    std::vector<std::int8_t> h_interactions(num_interactions * dimX * dimY * 2);
    std::vector<double> h_factor(total_walker, std::numbers::e);

    wl::device_ptr<double> d_factor(h_factor);
    initialize_randomly(h_lattice);
    initialize_randomly(h_interactions);
    std::vector<int> h_offset_lattice(total_walker, 0);
    for(std::size_t i = 0; i<num_interactions*num_intervals_per_interaction; i++){
        for(std::size_t j = 0; j < num_walker_per_interval; j++){
            h_offset_lattice[i*num_walker_per_interval+j] = (static_cast<int>(i)*num_walker_per_interval+static_cast<int>(j))*dimX*dimY;
        }
    }

    wl::device_ptr<std::int8_t> d_lattice(h_lattice);
    wl::device_ptr<std::int8_t> d_interactions(h_interactions);
    wl::device_ptr<int> d_offset_lattice(h_offset_lattice);

    wl::calc_energy(total_intervals,
                    num_walker_per_interval,
                    wl::boundary::periodic,
                    d_energy.data(),
                    d_lattice.data(),
                    d_interactions.data(),
                    d_offset_lattice.data(),
                    dimX,
                    dimY,
                    total_walker,
                    num_walker_per_interval*num_intervals_per_interaction);

    get_new_config_energy_and_flipped_spin_index_periodic<<<total_intervals, num_walker_per_interval>>>(d_new_config.data(), d_lattice.data(), d_interactions.data(), d_energy.data(), d_offset_iter.data(), d_offset_lattice.data(), dimX, dimY, total_walker, num_walker_per_interval*num_intervals_per_interaction, seed);

    std::vector<RBIM> h_new_config(total_walker);
    std::vector<std::int8_t> h_lattice_expected(total_walker * dimX * dimY);
    std::vector<unsigned long long> h_H_expected(total_len_histogram, 0);
    std::vector<double> h_log_g_expected(total_len_histogram, 0);
    d_new_config.device_to_host(h_new_config);
    d_lattice.device_to_host(h_lattice_expected);

    // Get final next spin configs and check the energies stored in the RBIM struct
    for(std::size_t i = 0; i < total_walker; i++){
        std::size_t spin_in_lattice_idx =  static_cast<std::size_t>(h_new_config[i].i) *dimY + static_cast<std::size_t>(h_new_config[i].j);
        int histogram_idx = h_new_config[i].new_energy+2*dimX*dimY;
        BOOST_CHECK_GE(histogram_idx, 0);
        h_lattice_expected[static_cast<std::size_t>(h_offset_lattice[i])+spin_in_lattice_idx] *= -1;
        h_H_expected[i*(4*dimX*dimY+1)+static_cast<std::size_t>(histogram_idx)]+=1;
        h_log_g_expected[i*(4*dimX*dimY+1)+static_cast<std::size_t>(histogram_idx)]+=1;
    }
    wl::device_ptr<std::int8_t> d_lattice_expected(total_walker * dimX * dimY, 0);
    wl::device_ptr<int> d_energy_expected(total_walker, 0);
    d_lattice_expected.host_to_device(h_lattice_expected);

    wl::calc_energy(total_intervals,
                    num_walker_per_interval,
                    wl::boundary::periodic,
                    d_energy_expected.data(),
                    d_lattice_expected.data(),
                    d_interactions.data(),
                    d_offset_lattice.data(),
                    dimX,
                    dimY,
                    total_walker,
                    num_walker_per_interval*num_intervals_per_interaction);

    std::vector<int> h_energy_expected(total_walker);
    d_energy_expected.device_to_host(h_energy_expected);
    for(std::size_t i = 0; i < total_walker; i++){
        BOOST_CHECK_EQUAL(h_energy_expected[i], h_new_config[i].new_energy);
    }

    std::vector<int> h_interval_start_energies(num_interactions*num_intervals_per_interaction);
    std::vector<int> h_interval_end_energies(num_interactions*num_intervals_per_interaction);
    std::vector<int> h_offset_histogram(total_walker);
    int incremental_offset_histogram = 0;
    for(std::size_t i = 0; i<num_interactions*num_intervals_per_interaction; i++){
        int interval_id = static_cast<int>(i) % num_intervals_per_interaction;
        int interaction_id = static_cast<int>(i) / num_interactions;
        h_interval_start_energies[i] = -2*dimX*dimY;
        h_interval_end_energies[i] = 2*dimX*dimY;
        for(std::size_t j = 0; j < num_walker_per_interval; j++){
            h_offset_histogram[static_cast<std::size_t>(interaction_id)*num_intervals_per_interaction*num_walker_per_interval+static_cast<std::size_t>(interval_id)*num_walker_per_interval+j] = incremental_offset_histogram;
            incremental_offset_histogram += h_interval_end_energies[i] - h_interval_start_energies[i] + 1;
        }
    }

    wl::device_ptr<int> d_start(h_interval_start_energies);
    wl::device_ptr<int> d_end(h_interval_end_energies);
    wl::device_ptr<int> d_offset_histogram(h_offset_histogram);
    wl::device_ptr<std::int8_t> d_expected_energy_spectrum(num_interactions*2*dimX*dimY+1, 1);
    wl::device_ptr<int> d_offset_energy_spectrum(num_interactions, 0);

    // Reset d_offset_iter
    std::vector<unsigned long> h_offset_iter(total_walker,0);
    d_offset_iter.host_to_device(h_offset_iter);

    // Execute a single Wang Landau step from initial config which shoudl result in all changes accpeted
    wl::wang_landau<<<total_intervals, num_walker_per_interval>>>(d_lattice.data(),
                                                                        d_interactions.data(),
                                                                        d_H.data(),
                                                                        d_logG.data(),
                                                                        d_factor.data(),
                                                                        d_energy.data(),
                                                                        d_newEnergies.data(),
                                                                        d_offset_iter.data(),
                                                                        d_foundNewEnergyFlag.data(),
                                                                        d_start.data(),
                                                                        d_end.data(),
                                                                        d_offset_histogram.data(),
                                                                        d_offset_lattice.data(),
                                                                        d_cond.data(),
                                                                        d_expected_energy_spectrum.data(),
                                                                        d_offset_energy_spectrum.data(),
                                                                        d_cond_interactions.data(),
                                                                        dimX,
                                                                        dimY,
                                                                        num_iterations,
                                                                        total_walker,
                                                                        num_intervals_per_interaction,
                                                                        num_walker_per_interval * num_intervals_per_interaction,
                                                                        seed,
                                                                        wl::boundary::periodic);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<std::int8_t> h_lattice_result(total_walker * dimX * dimY, 0);
    std::vector<unsigned long long> h_H_result(total_len_histogram, 0);
    std::vector<double> h_log_g_result(total_len_histogram, 0);
    d_lattice.device_to_host(h_lattice_result);
    d_logG.device_to_host(h_log_g_result);
    d_H.device_to_host(h_H_result);

    BOOST_CHECK_EQUAL_COLLECTIONS(h_lattice_result.begin(), h_lattice_result.end(), h_lattice_expected.begin(), h_lattice_expected.end());
    BOOST_REQUIRE_EQUAL(h_log_g_result.size(), h_log_g_expected.size());
    for(std::size_t i = 0; i<h_log_g_result.size(); i++){
        BOOST_CHECK_SMALL(h_log_g_result[i] - h_log_g_expected[i], 1e-15);
    }
    BOOST_CHECK_EQUAL_COLLECTIONS(h_H_result.begin(), h_H_result.end(), h_H_expected.begin(), h_H_expected.end());
}

BOOST_AUTO_TEST_CASE(test_wang_landau_open_success)
{
    const int num_interactions = 10;
    const int num_intervals_per_interaction = 10;
    const int num_walker_per_interval = 5;
    const int total_walker = num_interactions*num_intervals_per_interaction*num_walker_per_interval;
    const int total_intervals = num_intervals_per_interaction * num_interactions;
    const int seed = 42;
    const int dimX = 4;
    const int dimY = 4;
    const int num_iterations = 1;
    const int total_len_histogram = total_walker * 4*dimX*dimY + 1;


    // Device arrays
    wl::device_ptr<unsigned long> d_offset_iter(total_walker, 0);
    wl::device_ptr<int> d_energy(total_walker, 0);
    wl::device_ptr<RBIM> d_new_config(total_walker);
    wl::device_ptr<int> d_cond_interactions(num_interactions, 0);
    wl::device_ptr<std::int8_t> d_cond(total_intervals, 0);
    wl::device_ptr<int> d_newEnergies(total_walker);
    wl::device_ptr<int> d_foundNewEnergyFlag(total_walker);
    wl::device_ptr<unsigned long long> d_H(total_len_histogram, 0);
    wl::device_ptr<double> d_logG(total_len_histogram, 0);

    // Copy data from host to device
    std::vector<std::int8_t> h_lattice(total_walker * dimX * dimY);
    std::vector<std::int8_t> h_interactions(num_interactions * dimX * dimY * 2);
    std::vector<double> h_factor(total_walker, std::numbers::e);

    wl::device_ptr<double> d_factor(h_factor);
    initialize_randomly(h_lattice);
    initialize_randomly(h_interactions);
    std::vector<int> h_offset_lattice(total_walker, 0);
    for(std::size_t i = 0; i<num_interactions*num_intervals_per_interaction; i++){
        for(std::size_t j = 0; j < num_walker_per_interval; j++){
            h_offset_lattice[i*num_walker_per_interval+j] = (static_cast<int>(i)*num_walker_per_interval+static_cast<int>(j))*dimX*dimY;
        }
    }

    wl::device_ptr<std::int8_t> d_lattice(h_lattice);
    wl::device_ptr<std::int8_t> d_interactions(h_interactions);
    wl::device_ptr<int> d_offset_lattice(h_offset_lattice);

    wl::calc_energy(total_intervals,
                    num_walker_per_interval,
                    wl::boundary::open,
                    d_energy.data(),
                    d_lattice.data(),
                    d_interactions.data(),
                    d_offset_lattice.data(),
                    dimX,
                    dimY,
                    total_walker,
                    num_walker_per_interval*num_intervals_per_interaction);

    get_new_config_energy_and_flipped_spin_index_open<<<total_intervals, num_walker_per_interval>>>(d_new_config.data(), d_lattice.data(), d_interactions.data(), d_energy.data(), d_offset_iter.data(), d_offset_lattice.data(), dimX, dimY, total_walker, num_walker_per_interval*num_intervals_per_interaction, seed);

    std::vector<RBIM> h_new_config(total_walker);
    std::vector<std::int8_t> h_lattice_expected(total_walker * dimX * dimY);
    std::vector<unsigned long long> h_H_expected(total_len_histogram, 0);
    std::vector<double> h_log_g_expected(total_len_histogram, 0);
    d_new_config.device_to_host(h_new_config);
    d_lattice.device_to_host(h_lattice_expected);

    // Get final next spin configs and check the energies stored in the RBIM struct
    for(std::size_t i = 0; i < total_walker; i++){
        std::size_t spin_in_lattice_idx =  static_cast<std::size_t>(h_new_config[i].i) *dimY + static_cast<std::size_t>(h_new_config[i].j);
        int histogram_idx = h_new_config[i].new_energy+2*dimX*dimY;
        BOOST_CHECK_GE(histogram_idx, 0);
        h_lattice_expected[static_cast<std::size_t>(h_offset_lattice[i])+spin_in_lattice_idx] *= -1;
        h_H_expected[i*(4*dimX*dimY+1)+static_cast<std::size_t>(histogram_idx)]+=1;
        h_log_g_expected[i*(4*dimX*dimY+1)+static_cast<std::size_t>(histogram_idx)]+=1;
    }
    wl::device_ptr<std::int8_t> d_lattice_expected(total_walker * dimX * dimY, 0);
    wl::device_ptr<int> d_energy_expected(total_walker, 0);
    d_lattice_expected.host_to_device(h_lattice_expected);

    wl::calc_energy(total_intervals,
                    num_walker_per_interval,
                    wl::boundary::open,
                    d_energy_expected.data(),
                    d_lattice_expected.data(),
                    d_interactions.data(),
                    d_offset_lattice.data(),
                    dimX,
                    dimY,
                    total_walker,
                    num_walker_per_interval*num_intervals_per_interaction);

    std::vector<int> h_energy_expected(total_walker);
    d_energy_expected.device_to_host(h_energy_expected);
    for(std::size_t i = 0; i < total_walker; i++){
        BOOST_CHECK_EQUAL(h_energy_expected[i], h_new_config[i].new_energy);
    }

    std::vector<int> h_interval_start_energies(num_interactions*num_intervals_per_interaction);
    std::vector<int> h_interval_end_energies(num_interactions*num_intervals_per_interaction);
    std::vector<int> h_offset_histogram(total_walker);
    int incremental_offset_histogram = 0;
    for(std::size_t i = 0; i<num_interactions*num_intervals_per_interaction; i++){
        int interval_id = static_cast<int>(i) % num_intervals_per_interaction;
        int interaction_id = static_cast<int>(i) / num_interactions;
        h_interval_start_energies[i] = -2*dimX*dimY;
        h_interval_end_energies[i] = 2*dimX*dimY;
        for(std::size_t j = 0; j < num_walker_per_interval; j++){
            h_offset_histogram[static_cast<std::size_t>(interaction_id)*num_intervals_per_interaction*num_walker_per_interval+static_cast<std::size_t>(interval_id)*num_walker_per_interval+j] = incremental_offset_histogram;
            incremental_offset_histogram += h_interval_end_energies[i] - h_interval_start_energies[i] + 1;
        }
    }

    wl::device_ptr<int> d_start(h_interval_start_energies);
    wl::device_ptr<int> d_end(h_interval_end_energies);
    wl::device_ptr<int> d_offset_histogram(h_offset_histogram);
    wl::device_ptr<std::int8_t> d_expected_energy_spectrum(num_interactions*2*dimX*dimY+1, 1);
    wl::device_ptr<int> d_offset_energy_spectrum(num_interactions, 0);

    // Reset d_offset_iter
    std::vector<unsigned long> h_offset_iter(total_walker,0);
    d_offset_iter.host_to_device(h_offset_iter);

    // Execute a single Wang Landau step from initial config which shoudl result in all changes accpeted
    wl::wang_landau<<<total_intervals, num_walker_per_interval>>>(d_lattice.data(),
                                                                        d_interactions.data(),
                                                                        d_H.data(),
                                                                        d_logG.data(),
                                                                        d_factor.data(),
                                                                        d_energy.data(),
                                                                        d_newEnergies.data(),
                                                                        d_offset_iter.data(),
                                                                        d_foundNewEnergyFlag.data(),
                                                                        d_start.data(),
                                                                        d_end.data(),
                                                                        d_offset_histogram.data(),
                                                                        d_offset_lattice.data(),
                                                                        d_cond.data(),
                                                                        d_expected_energy_spectrum.data(),
                                                                        d_offset_energy_spectrum.data(),
                                                                        d_cond_interactions.data(),
                                                                        dimX,
                                                                        dimY,
                                                                        num_iterations,
                                                                        total_walker,
                                                                        num_intervals_per_interaction,
                                                                        num_walker_per_interval * num_intervals_per_interaction,
                                                                        seed,
                                                                        wl::boundary::open);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<std::int8_t> h_lattice_result(total_walker * dimX * dimY, 0);
    std::vector<unsigned long long> h_H_result(total_len_histogram, 0);
    std::vector<double> h_log_g_result(total_len_histogram, 0);
    d_lattice.device_to_host(h_lattice_result);
    d_logG.device_to_host(h_log_g_result);
    d_H.device_to_host(h_H_result);

    BOOST_CHECK_EQUAL_COLLECTIONS(h_lattice_result.begin(), h_lattice_result.end(), h_lattice_expected.begin(), h_lattice_expected.end());
    BOOST_REQUIRE_EQUAL(h_log_g_result.size(), h_log_g_expected.size());
    for(std::size_t i = 0; i<h_log_g_result.size(); i++){
        BOOST_CHECK_SMALL(h_log_g_result[i] - h_log_g_expected[i], 1e-15);
    }
    BOOST_CHECK_EQUAL_COLLECTIONS(h_H_result.begin(), h_H_result.end(), h_H_expected.begin(), h_H_expected.end());
}

BOOST_AUTO_TEST_SUITE_END()
