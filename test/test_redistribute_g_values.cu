#include <boost/test/tools/old/interface.hpp>
#include <cstddef>
#define BOOST_TEST_MODULE wl_tests

#include <boost/test/unit_test.hpp>
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <cmath>

#include "../src/wl.cuh"
#include "../src/utils.hpp"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            BOOST_FAIL("CUDA Error: " << cudaGetErrorString(err)); \
        } \
    } while (0)

BOOST_AUTO_TEST_SUITE(redistribute_g_values)

BOOST_AUTO_TEST_CASE(test_redistribute_g_values_success)
{
    const int num_interactions = 3;
    const int num_intervals_per_interaction = 3;
    const int num_walker_per_interval = 4;
    const int total_len_histogram = 20*4;

    // Host arrays
    std::vector<double> h_log_G(total_len_histogram, 0.0);
    std::vector<double> h_shared_logG = {1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.10, 11.11, 12.12, 13.13, 14.14, 15.15, 16.16, 17.17, 18.18, 19.19, 20.20};
    std::vector<int> h_len_histograms = {7*4, 9*4, 4*4};
    std::vector<int> h_interval_start_energies = {-2, -1, 0, -4, -2, 0, -2, -1, 0};
    std::vector<int> h_interval_end_energies = {-1, 0, 2, -2, 0, 2, -2, -1, 1};
    std::vector<int> h_cond_interaction = {1, 1, 0};
    std::vector<int> h_offset_histogram = {
        0, 2, 2*2, 3*2,
        4*2 , 5*2, 6*2,7*2,
        8*2, 8*2+3, 8*2+2*3, 8*2+3*3,
        8*2+4*3, 8*2+5*3, 8*2+6*3, 8*2+7*3,
        8*2+8*3, 8*2+9*3, 8*2+10*3, 8*2+11*3,
        8*2+12*3, 8*2+13*3, 8*2+14*3, 8*2+15*3,
        8*2+16*3,8*2+16*3+1,8*2+16*3+2,8*2+16*3+3,
        8*2+16*3+4,8*2+16*3+5, 8*2+16*3+6,8*2+16*3+7,
        8*2+16*3+8, 8*2+16*3+8+2,8*2+16*3+8+2*2, 8*2+16*3+8+3*2
    };
    std::vector<int8_t> h_cond_interval = {1, 1, 1, 1, 1, 0, 1, 0,0};
    std::vector<long long> h_offset_shared_logG = {0, 2, 4, 7, 10, 13, 16, 17, 18};

    // Device arrays
    double* d_log_G;
    double* d_shared_logG;
    int* d_len_histograms;
    int* d_interval_start_energies;
    int* d_interval_end_energies;
    int* d_cond_interaction;
    int* d_offset_histogram;
    int8_t* d_cond_interval;
    long long* d_offset_shared_logG;

    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&d_log_G, total_len_histogram * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_shared_logG, h_shared_logG.size() * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_len_histograms, h_len_histograms.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_interval_start_energies, h_interval_start_energies.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_interval_end_energies, h_interval_end_energies.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cond_interaction, h_cond_interaction.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_offset_histogram, h_offset_histogram.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cond_interval, h_cond_interval.size() * sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(&d_offset_shared_logG, h_offset_shared_logG.size() * sizeof(long long)));

    CUDA_CHECK(cudaMemset(d_log_G, 0, total_len_histogram * sizeof(double)));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_shared_logG, h_shared_logG.data(), h_shared_logG.size() * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_len_histograms, h_len_histograms.data(), h_len_histograms.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_interval_start_energies, h_interval_start_energies.data(), h_interval_start_energies.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_interval_end_energies, h_interval_end_energies.data(), h_interval_end_energies.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cond_interaction, h_cond_interaction.data(), h_cond_interaction.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_offset_histogram, h_offset_histogram.data(), h_offset_histogram.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cond_interval, h_cond_interval.data(), h_cond_interval.size() * sizeof(int8_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_offset_shared_logG, h_offset_shared_logG.data(), h_offset_shared_logG.size() * sizeof(long long), cudaMemcpyHostToDevice));

    // Launch kernel
    const int max_threads_per_block = 128;
    const int block_count = (total_len_histogram + max_threads_per_block - 1) / max_threads_per_block;

    wl::redistribute_g_values<<<block_count, max_threads_per_block>>>(
        d_log_G, d_shared_logG, d_len_histograms, d_interval_start_energies, d_interval_end_energies,
        d_cond_interaction, d_offset_histogram, d_cond_interval, d_offset_shared_logG,
        num_intervals_per_interaction, num_walker_per_interval, num_interactions, total_len_histogram);

    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy results back to host
    CUDA_CHECK(cudaMemcpy(h_log_G.data(), d_log_G, h_log_G.size() * sizeof(double), cudaMemcpyDeviceToHost));

    // Expectation vector
    std::vector<double> expected_log_G = {1.1, 2.2,1.1, 2.2,1.1, 2.2,1.1, 2.2, 3.3, 4.4, 3.3,
     4.4, 3.3, 4.4, 3.3, 4.4, 5.5, 6.6, 7.7, 5.5,
     6.6, 7.7,5.5, 6.6, 7.7,5.5, 6.6, 7.7,8.8,
     9.9, 10.10,8.8, 9.9, 10.10,8.8, 9.9, 10.10,8.8,
     9.9, 10.10, 11.11, 12.12, 13.13,11.11, 12.12, 13.13,11.11,
     12.12, 13.13 ,11.11, 12.12, 13.13, 0,0,0,0,
     0,0,0,0,0,0,0,0, 17.17,
     17.17,17.17,17.17,0,0,0,0,0,0,
     0,0,0,0,0,0
    };

    // checks
    for (size_t i = 0; i < h_log_G.size(); ++i) {
        BOOST_CHECK_CLOSE(h_log_G[i], expected_log_G[i], 1e-15);
    }

    // Free device memory
    CUDA_CHECK(cudaFree(d_log_G));
    CUDA_CHECK(cudaFree(d_shared_logG));
    CUDA_CHECK(cudaFree(d_len_histograms));
    CUDA_CHECK(cudaFree(d_interval_start_energies));
    CUDA_CHECK(cudaFree(d_interval_end_energies));
    CUDA_CHECK(cudaFree(d_cond_interaction));
    CUDA_CHECK(cudaFree(d_offset_histogram));
    CUDA_CHECK(cudaFree(d_cond_interval));
    CUDA_CHECK(cudaFree(d_offset_shared_logG));
}


BOOST_AUTO_TEST_SUITE_END()