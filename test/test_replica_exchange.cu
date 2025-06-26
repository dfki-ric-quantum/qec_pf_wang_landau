#include <boost/test/tools/old/interface.hpp>
#define BOOST_TEST_MODULE wl_tests

#include <cstddef>
#include <boost/test/unit_test.hpp>
#include <cuda_runtime.h>
#include <vector>

#include "../src/wl.cuh"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            BOOST_FAIL("CUDA Error: " << cudaGetErrorString(err)); \
        } \
    } while (0)

__global__ void fisher_yates_wrapper(int* d_indices,
                             unsigned long* d_offset_iter,
                             int seed,
                             int walker_per_interactions) {
    unsigned long tid = blockDim.x * blockIdx.x + threadIdx.x;
    const int interaction_id = static_cast<int>(tid) / walker_per_interactions;
    unsigned long swap_id = static_cast<long long>(blockDim.x) * (blockIdx.x + 1);
    if (threadIdx.x == 0) {
        wl::fisher_yates(d_indices, d_offset_iter, seed, interaction_id, walker_per_interactions);
    }
    __syncthreads();
    swap_id += d_indices[tid];
}


BOOST_AUTO_TEST_SUITE(replica_exchange)

BOOST_AUTO_TEST_CASE(test_replic_exchange_even_success)
{
    const int num_interactions = 10;
    const int num_intervals_per_interaction = 10;
    const int num_walker_per_interval = 5;
    const int total_walker = num_interactions*num_intervals_per_interaction*num_walker_per_interval;
    const int seed = 42;
    const int lattice_dim = 9;

    int total_len_histogram = 0;
    int incremental_offset_histogram = 0;
    std::vector<int> h_offset_histogram(total_walker);
    std::vector<int> h_interval_start_energies(num_interactions*num_intervals_per_interaction);
    std::vector<int> h_interval_end_energies(num_interactions*num_intervals_per_interaction);
    std::vector<int> h_energy(total_walker);
    std::vector<int> h_offset_lattice(total_walker);
    for(std::size_t i = 0; i<num_interactions*num_intervals_per_interaction; i++){
        int interval_id = static_cast<int>(i) % num_intervals_per_interaction;
        int interaction_id = static_cast<int>(i) / num_interactions;
        h_interval_start_energies[i] = interval_id + interaction_id;
        h_interval_end_energies[i] = interval_id + interaction_id + 4;
        total_len_histogram += num_walker_per_interval * (h_interval_end_energies[i]-h_interval_start_energies[i]);
        for(std::size_t j = 0; j < num_walker_per_interval; j++){
            h_energy[static_cast<std::size_t>(interaction_id)*num_intervals_per_interaction*num_walker_per_interval+static_cast<std::size_t>(interval_id)*num_walker_per_interval+j] = interval_id + interaction_id + 2;
            h_offset_histogram[static_cast<std::size_t>(interaction_id)*num_intervals_per_interaction*num_walker_per_interval+static_cast<std::size_t>(interval_id)*num_walker_per_interval+j] = incremental_offset_histogram;
            incremental_offset_histogram += h_interval_end_energies[i]-h_interval_start_energies[i];
            size_t offset_lattice = (i * num_walker_per_interval + j) * lattice_dim;
            h_offset_lattice[i * num_walker_per_interval + j] = static_cast<int>(offset_lattice);
        }
    }
    std::vector<double> h_log_G(total_len_histogram, 1.0); // flat log(g) array should result in accept switch all the time
    std::vector<int> h_cond_interaction = {1, 1, 1}; // if interaction carries -1 the replica exchange wont be executed
    std::vector<unsigned long> h_offset_iter(total_walker, 0);

    // Device pointer
    double* d_log_G;
    int* d_interval_start_energies;
    int* d_interval_end_energies;
    int* d_cond_interaction;
    int* d_offset_histogram;
    int* d_energy;
    int* d_offset_lattice;
    wl::device_ptr<unsigned long> d_offset_iter(total_walker, 0);
    wl::device_ptr<int> d_indices(total_walker);

    CUDA_CHECK(cudaMalloc(&d_log_G, total_len_histogram * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_interval_start_energies, h_interval_start_energies.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_interval_end_energies, h_interval_end_energies.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cond_interaction, h_cond_interaction.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_offset_histogram, h_offset_histogram.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_energy, h_energy.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_offset_lattice, h_offset_lattice.size() * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_log_G, h_log_G.data(), h_log_G.size() * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_interval_start_energies, h_interval_start_energies.data(), h_interval_start_energies.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_interval_end_energies, h_interval_end_energies.data(), h_interval_end_energies.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cond_interaction, h_cond_interaction.data(), h_cond_interaction.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_offset_histogram, h_offset_histogram.data(), h_offset_histogram.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_energy, h_energy.data(), h_energy.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_offset_lattice, h_offset_lattice.data(), h_offset_lattice.size() * sizeof(int), cudaMemcpyHostToDevice));

    // Expectations are deduced by statistically generated permutation captured as index set in d_indices - These will be used to compare the lattice offsets before and after the replica exchange execution
    wl::init_indices<<<num_interactions*num_intervals_per_interaction, num_walker_per_interval>>>(d_indices.data(), total_walker);
    fisher_yates_wrapper<<<num_intervals_per_interaction * num_interactions, num_walker_per_interval>>>(d_indices.data(), d_offset_iter.data(), seed, num_walker_per_interval*num_intervals_per_interaction);
    std::vector<int> h_indices_expected(total_walker);
    d_indices.device_to_host(h_indices_expected);
    for(std::size_t i = 0; i < num_interactions*num_intervals_per_interaction; i++){
        int index_sum = 0;
        for(std::size_t j = 0; j < num_walker_per_interval; j++){
            index_sum += h_indices_expected[i*num_walker_per_interval + j];
        }
        // minor sanity check
        BOOST_CHECK_EQUAL(index_sum, num_walker_per_interval*(num_walker_per_interval-1)/2);
    }

    // Reinitialization of indices and iterator offsets for deterministic run of replica exchange
    wl::init_indices<<<num_interactions*num_intervals_per_interaction, num_walker_per_interval>>>(d_indices.data(), total_walker);
    d_offset_iter.host_to_device(h_offset_iter);

    // Launch kernel
    wl::replica_exchange<<<num_intervals_per_interaction * num_interactions, num_walker_per_interval>>>(d_offset_lattice,
                                                                            d_energy,
                                                                            d_indices.data(),
                                                                            d_offset_iter.data(),
                                                                            d_interval_start_energies,
                                                                            d_interval_end_energies,
                                                                            d_offset_histogram,
                                                                            d_cond_interaction,
                                                                            d_log_G,
                                                                            true,
                                                                            seed,
                                                                            num_intervals_per_interaction,
                                                                            num_intervals_per_interaction * num_walker_per_interval);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Get results for checks
    std::vector<int> h_offset_lattice_results(total_walker);
    std::vector<int> h_energy_results(total_walker);
    CUDA_CHECK(cudaMemcpy(h_offset_lattice_results.data(), d_offset_lattice, h_offset_lattice.size() * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_energy_results.data(), d_energy, h_energy_results.size() * sizeof(int), cudaMemcpyDeviceToHost));

    // Checks
    for(std::size_t i = 0; i < num_interactions*num_intervals_per_interaction; i++){
        int interval_id = static_cast<int>(i) % num_intervals_per_interaction;
        for(std::size_t j = 0; j < num_walker_per_interval; j++){
            if (interval_id%2==0 && interval_id+1<num_intervals_per_interaction){
                BOOST_CHECK_EQUAL(h_offset_lattice[(i+1)*num_walker_per_interval+static_cast<std::size_t>(h_indices_expected[i*num_walker_per_interval+j])], h_offset_lattice_results[i*num_walker_per_interval+j]);
                BOOST_CHECK_EQUAL(h_offset_lattice[i*num_walker_per_interval+j], h_offset_lattice_results[(i+1)*num_walker_per_interval+static_cast<std::size_t>(h_indices_expected[i*num_walker_per_interval+j])]);

                BOOST_CHECK_EQUAL(h_energy[(i+1)*num_walker_per_interval+static_cast<std::size_t>(h_indices_expected[i*num_walker_per_interval+j])], h_energy_results[i*num_walker_per_interval+j]);
                BOOST_CHECK_EQUAL(h_energy[i*num_walker_per_interval+j], h_energy_results[(i+1)*num_walker_per_interval+static_cast<std::size_t>(h_indices_expected[i*num_walker_per_interval+j])]);
            }
        }
    }

    // Free device memory
    CUDA_CHECK(cudaFree(d_interval_start_energies));
    CUDA_CHECK(cudaFree(d_interval_end_energies));
    CUDA_CHECK(cudaFree(d_offset_histogram));
    CUDA_CHECK(cudaFree(d_cond_interaction));
    CUDA_CHECK(cudaFree(d_log_G));
    CUDA_CHECK(cudaFree(d_energy));
}

BOOST_AUTO_TEST_CASE(test_replic_exchange_odd_success)
{
    const int num_interactions = 10;
    const int num_intervals_per_interaction = 10;
    const int num_walker_per_interval = 5;
    const int total_walker = num_interactions*num_intervals_per_interaction*num_walker_per_interval;
    const int seed = 42;
    const int lattice_dim = 9;

    int total_len_histogram = 0;
    int incremental_offset_histogram = 0;
    std::vector<int> h_offset_histogram(total_walker);
    std::vector<int> h_interval_start_energies(num_interactions*num_intervals_per_interaction);
    std::vector<int> h_interval_end_energies(num_interactions*num_intervals_per_interaction);
    std::vector<int> h_energy(total_walker);
    std::vector<int> h_offset_lattice(total_walker);
    for(std::size_t i = 0; i<num_interactions*num_intervals_per_interaction; i++){
        int interval_id = static_cast<int>(i) % num_intervals_per_interaction;
        int interaction_id = static_cast<int>(i) / num_interactions;
        h_interval_start_energies[i] = interval_id + interaction_id;
        h_interval_end_energies[i] = interval_id + interaction_id + 4;
        total_len_histogram += num_walker_per_interval * (h_interval_end_energies[i]-h_interval_start_energies[i]);
        for(std::size_t j = 0; j < num_walker_per_interval; j++){
            h_energy[static_cast<std::size_t>(interaction_id)*num_intervals_per_interaction*num_walker_per_interval+static_cast<std::size_t>(interval_id)*num_walker_per_interval+j] = interval_id + interaction_id + 2;
            h_offset_histogram[static_cast<std::size_t>(interaction_id)*num_intervals_per_interaction*num_walker_per_interval+static_cast<std::size_t>(interval_id)*num_walker_per_interval+j] = incremental_offset_histogram;
            incremental_offset_histogram += h_interval_end_energies[i]-h_interval_start_energies[i];
            size_t offset_lattice = (i * num_walker_per_interval + j) * lattice_dim;
            h_offset_lattice[i * num_walker_per_interval + j] = static_cast<int>(offset_lattice);
        }
    }
    std::vector<double> h_log_G(total_len_histogram, 1.0); // flat log(g) array should result in accept switch all the time
    std::vector<int> h_cond_interaction = {1, 1, 1}; // if interaction carries -1 the replica exchange wont be executed
    std::vector<unsigned long> h_offset_iter(total_walker, 0);

    // Device pointer
    double* d_log_G;
    int* d_interval_start_energies;
    int* d_interval_end_energies;
    int* d_cond_interaction;
    int* d_offset_histogram;
    int* d_energy;
    int* d_offset_lattice;
    wl::device_ptr<unsigned long> d_offset_iter(total_walker, 0);
    wl::device_ptr<int> d_indices(total_walker);

    CUDA_CHECK(cudaMalloc(&d_log_G, total_len_histogram * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_interval_start_energies, h_interval_start_energies.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_interval_end_energies, h_interval_end_energies.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cond_interaction, h_cond_interaction.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_offset_histogram, h_offset_histogram.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_energy, h_energy.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_offset_lattice, h_offset_lattice.size() * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_log_G, h_log_G.data(), h_log_G.size() * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_interval_start_energies, h_interval_start_energies.data(), h_interval_start_energies.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_interval_end_energies, h_interval_end_energies.data(), h_interval_end_energies.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cond_interaction, h_cond_interaction.data(), h_cond_interaction.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_offset_histogram, h_offset_histogram.data(), h_offset_histogram.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_energy, h_energy.data(), h_energy.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_offset_lattice, h_offset_lattice.data(), h_offset_lattice.size() * sizeof(int), cudaMemcpyHostToDevice));

    // Expectations are deduced by statistically generated permutation captured as index set in d_indices - These will be used to compare the lattice offsets before and after the replica exchange execution
    wl::init_indices<<<num_interactions*num_intervals_per_interaction, num_walker_per_interval>>>(d_indices.data(), total_walker);
    fisher_yates_wrapper<<<num_intervals_per_interaction * num_interactions, num_walker_per_interval>>>(d_indices.data(), d_offset_iter.data(), seed, num_walker_per_interval*num_intervals_per_interaction);
    std::vector<int> h_indices_expected(total_walker);
    d_indices.device_to_host(h_indices_expected);
    for(std::size_t i = 0; i < num_interactions*num_intervals_per_interaction; i++){
        int index_sum = 0;
        for(std::size_t j = 0; j < num_walker_per_interval; j++){
            index_sum += h_indices_expected[i*num_walker_per_interval + j];
        }
        // minor sanity check
        BOOST_CHECK_EQUAL(index_sum, num_walker_per_interval*(num_walker_per_interval-1)/2);
    }

    // Reinitialization of indices and iterator offsets for deterministic run of replica exchange
    wl::init_indices<<<num_interactions*num_intervals_per_interaction, num_walker_per_interval>>>(d_indices.data(), total_walker);
    d_offset_iter.host_to_device(h_offset_iter);

    // Launch kernel
    wl::replica_exchange<<<num_intervals_per_interaction * num_interactions, num_walker_per_interval>>>(d_offset_lattice,
                                                                            d_energy,
                                                                            d_indices.data(),
                                                                            d_offset_iter.data(),
                                                                            d_interval_start_energies,
                                                                            d_interval_end_energies,
                                                                            d_offset_histogram,
                                                                            d_cond_interaction,
                                                                            d_log_G,
                                                                            false,
                                                                            seed,
                                                                            num_intervals_per_interaction,
                                                                            num_intervals_per_interaction * num_walker_per_interval);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Get results for checks
    std::vector<int> h_offset_lattice_results(total_walker);
    std::vector<int> h_energy_results(total_walker);
    CUDA_CHECK(cudaMemcpy(h_offset_lattice_results.data(), d_offset_lattice, h_offset_lattice.size() * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_energy_results.data(), d_energy, h_energy_results.size() * sizeof(int), cudaMemcpyDeviceToHost));

    // Checks
    for(std::size_t i = 0; i < num_interactions*num_intervals_per_interaction; i++){
        int interval_id = static_cast<int>(i) % num_intervals_per_interaction;
        for(std::size_t j = 0; j < num_walker_per_interval; j++){
            if (interval_id%2!=0 && interval_id+1<num_intervals_per_interaction){
                BOOST_CHECK_EQUAL(h_offset_lattice[(i+1)*num_walker_per_interval+static_cast<std::size_t>(h_indices_expected[i*num_walker_per_interval+j])], h_offset_lattice_results[i*num_walker_per_interval+j]);
                BOOST_CHECK_EQUAL(h_offset_lattice[i*num_walker_per_interval+j], h_offset_lattice_results[(i+1)*num_walker_per_interval+static_cast<std::size_t>(h_indices_expected[i*num_walker_per_interval+j])]);

                BOOST_CHECK_EQUAL(h_energy[(i+1)*num_walker_per_interval+static_cast<std::size_t>(h_indices_expected[i*num_walker_per_interval+j])], h_energy_results[i*num_walker_per_interval+j]);
                BOOST_CHECK_EQUAL(h_energy[i*num_walker_per_interval+j], h_energy_results[(i+1)*num_walker_per_interval+static_cast<std::size_t>(h_indices_expected[i*num_walker_per_interval+j])]);
            }
        }
    }

    // Free device memory
    CUDA_CHECK(cudaFree(d_interval_start_energies));
    CUDA_CHECK(cudaFree(d_interval_end_energies));
    CUDA_CHECK(cudaFree(d_offset_histogram));
    CUDA_CHECK(cudaFree(d_cond_interaction));
    CUDA_CHECK(cudaFree(d_log_G));
    CUDA_CHECK(cudaFree(d_energy));
}

BOOST_AUTO_TEST_SUITE_END()