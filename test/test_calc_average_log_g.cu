#include <boost/test/tools/old/interface.hpp>
#define BOOST_TEST_MODULE wl_tests

#include <cstddef>
#include <boost/test/unit_test.hpp>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>

#include "../src/wl.cuh"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            BOOST_FAIL("CUDA Error: " << cudaGetErrorString(err)); \
        } \
    } while (0)

std::size_t find_index(const std::vector<int>& h_offset_histogram, std::size_t value_a) {
    // Use std::lower_bound to find the first index where h_offset_histogram[i] >= value_a
    auto it = std::lower_bound(h_offset_histogram.begin(), h_offset_histogram.end(), value_a);

    if (it != h_offset_histogram.end()) {
        // distance must be positive as it comes after or equal h_offset_histogram.begin()
        auto i = static_cast<std::size_t>(std::distance(h_offset_histogram.begin(), it));

        // h_offset_histogram is positive such that cast wont affect value
        if (i == 0 || value_a > static_cast<std::size_t>(h_offset_histogram[i - 1])) {
            return i;
        }
    }

    throw std::out_of_range("value_a does not fall within any range.");
}

BOOST_AUTO_TEST_SUITE(calc_average_log_g)

BOOST_AUTO_TEST_CASE(test_calc_average_log_g_success)
{
    const int num_interactions = 3;
    const int num_intervals_per_interaction = 3;
    const int num_walker_per_interval = 4;
    const int total_len_histogram = 20*4;

    // Input arrays
    std::vector<double> h_log_G(total_len_histogram, 1.0);
    for(std::size_t i = 0; i < total_len_histogram; i++){
        h_log_G[i] = static_cast<double>(i);
    }
    std::vector<double> h_shared_log_G(20, 0.0);
    std::vector<int> h_len_histograms = {7*4, 9*4, 4*4};
    std::vector<int> h_interval_start_energies = {-2, -1, 0, -4, -2, 0, -2, -1, 0};
    std::vector<int> h_interval_end_energies = {-1, 0, 2, -2, 0, 2, -2, -1, 1};
    std::vector<int> h_cond_interaction = {1, 1, 1};
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
    std::vector<int8_t> h_cond_interval = {1, 1, 1, 1, 1, 1, 1, 1,1};
    std::vector<long long> h_offset_shared_logG = {0, 2, 4, 7, 10, 13, 16, 17, 18};
    std::vector<int> h_offset_energy_spectrum = {0, 7, 16};
    std::vector<std::int8_t> h_expected_energy_spectrum(20, 1);

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
    int8_t* d_expected_energy_spectrum;
    int* d_offset_energy_spectrum;

    CUDA_CHECK(cudaMalloc(&d_log_G, total_len_histogram * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_shared_logG, h_shared_log_G.size() * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_len_histograms, h_len_histograms.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_interval_start_energies, h_interval_start_energies.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_interval_end_energies, h_interval_end_energies.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cond_interaction, h_cond_interaction.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_offset_histogram, h_offset_histogram.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_offset_energy_spectrum, h_offset_energy_spectrum.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cond_interval, h_cond_interval.size() * sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(&d_offset_shared_logG, h_offset_shared_logG.size() * sizeof(long long)));
    CUDA_CHECK(cudaMalloc(&d_expected_energy_spectrum, h_expected_energy_spectrum.size() * sizeof(std::int8_t)));

    CUDA_CHECK(cudaMemset(d_shared_logG, 0, h_shared_log_G.size() * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(d_log_G, h_log_G.data(), h_log_G.size() * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_len_histograms, h_len_histograms.data(), h_len_histograms.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_interval_start_energies, h_interval_start_energies.data(), h_interval_start_energies.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_interval_end_energies, h_interval_end_energies.data(), h_interval_end_energies.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cond_interaction, h_cond_interaction.data(), h_cond_interaction.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_offset_histogram, h_offset_histogram.data(), h_offset_histogram.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_offset_energy_spectrum, h_offset_energy_spectrum.data(), h_offset_energy_spectrum.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cond_interval, h_cond_interval.data(), h_cond_interval.size() * sizeof(int8_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_offset_shared_logG, h_offset_shared_logG.data(), h_offset_shared_logG.size() * sizeof(long long), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_expected_energy_spectrum, h_expected_energy_spectrum.data(), h_expected_energy_spectrum.size() * sizeof(int8_t), cudaMemcpyHostToDevice));

    // Launch kernel
    const int max_threads_per_block = 128;
    const int block_count = (total_len_histogram + max_threads_per_block - 1) / max_threads_per_block;

    wl::calc_average_log_g<<<block_count, max_threads_per_block>>>(
        d_shared_logG, d_len_histograms, d_log_G, d_interval_start_energies, d_interval_end_energies, d_expected_energy_spectrum,
        d_cond_interval,  d_offset_histogram, d_offset_energy_spectrum, d_offset_shared_logG, d_cond_interaction,
        num_interactions, num_walker_per_interval, num_intervals_per_interaction, total_len_histogram);

    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy results back to host
    CUDA_CHECK(cudaMemcpy(h_shared_log_G.data(), d_shared_logG, h_shared_log_G.size() * sizeof(double), cudaMemcpyDeviceToHost));

    // Expectation
    std::vector<double> expected_h_shared_log_G(20, 0.0);
    for(std::size_t i = 0; i < total_len_histogram; i++){
        auto iterator = std::lower_bound(h_offset_histogram.begin(), h_offset_histogram.end(), i);
        // index is positive as iterator must be greater equal begin of histogram offsets
        std::size_t index = static_cast<std::size_t>(iterator - h_offset_histogram.begin());
        std::size_t walker_over_all_disorders_id = (iterator != h_offset_histogram.begin() && static_cast<std::size_t>(*iterator) != i) ? index - 1 : index;
        std::size_t interval_over_all_disorders_id = walker_over_all_disorders_id / num_walker_per_interval;
        if (walker_over_all_disorders_id >= h_offset_histogram.size()) {
            throw std::out_of_range("walker_over_all_disorders_id is out of range");
        }
        std::size_t energy_in_histogram =  i - static_cast<std::size_t>(h_offset_histogram[walker_over_all_disorders_id]);
        if(interval_over_all_disorders_id >= num_intervals_per_interaction * num_interactions){
            throw std::out_of_range("interval_over_all_disorders_id is out of range");
        }
        expected_h_shared_log_G[static_cast<std::size_t>(h_offset_shared_logG[interval_over_all_disorders_id])+energy_in_histogram] += h_log_G[i]/static_cast<double>(num_walker_per_interval);
    }

    // checks
    for (size_t i = 0; i < h_shared_log_G.size(); ++i) {
        BOOST_CHECK_CLOSE(h_shared_log_G[i], expected_h_shared_log_G[i], 1e-15);
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
    CUDA_CHECK(cudaFree(d_expected_energy_spectrum));
    CUDA_CHECK(cudaFree(d_offset_energy_spectrum));
}

BOOST_AUTO_TEST_CASE(test_calc_average_log_g_not_computed_for_non_ready_intervals)
{
    const int num_interactions = 3;
    const int num_intervals_per_interaction = 3;
    const int num_walker_per_interval = 4;
    const int total_len_histogram = 20*4;

    // Input arrays
    std::vector<double> h_log_G(total_len_histogram, 1.0);
    for(std::size_t i = 0; i < total_len_histogram; i++){
        h_log_G[i] = static_cast<double>(i);
    }
    std::vector<double> h_shared_log_G(20, 0.0);
    std::vector<int> h_len_histograms = {7*4, 9*4, 4*4};
    std::vector<int> h_interval_start_energies = {-2, -1, 0, -4, -2, 0, -2, -1, 0};
    std::vector<int> h_interval_end_energies = {-1, 0, 2, -2, 0, 2, -2, -1, 1};
    std::vector<int> h_cond_interaction = {1, 1, 1};
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
    std::vector<int8_t> h_cond_interval = {0, 0, 0, 0, 0, 1, 0, 1,1};
    std::vector<long long> h_offset_shared_logG = {0, 2, 4, 7, 10, 13, 16, 17, 18};
    std::vector<int> h_offset_energy_spectrum = {0, 7, 16};
    std::vector<std::int8_t> h_expected_energy_spectrum(20, 1);

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
    int8_t* d_expected_energy_spectrum;
    int* d_offset_energy_spectrum;

    CUDA_CHECK(cudaMalloc(&d_log_G, total_len_histogram * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_shared_logG, h_shared_log_G.size() * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_len_histograms, h_len_histograms.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_interval_start_energies, h_interval_start_energies.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_interval_end_energies, h_interval_end_energies.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cond_interaction, h_cond_interaction.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_offset_histogram, h_offset_histogram.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_offset_energy_spectrum, h_offset_energy_spectrum.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cond_interval, h_cond_interval.size() * sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(&d_offset_shared_logG, h_offset_shared_logG.size() * sizeof(long long)));
    CUDA_CHECK(cudaMalloc(&d_expected_energy_spectrum, h_expected_energy_spectrum.size() * sizeof(std::int8_t)));

    CUDA_CHECK(cudaMemset(d_shared_logG, 0, h_shared_log_G.size() * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(d_log_G, h_log_G.data(), h_log_G.size() * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_len_histograms, h_len_histograms.data(), h_len_histograms.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_interval_start_energies, h_interval_start_energies.data(), h_interval_start_energies.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_interval_end_energies, h_interval_end_energies.data(), h_interval_end_energies.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cond_interaction, h_cond_interaction.data(), h_cond_interaction.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_offset_histogram, h_offset_histogram.data(), h_offset_histogram.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_offset_energy_spectrum, h_offset_energy_spectrum.data(), h_offset_energy_spectrum.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cond_interval, h_cond_interval.data(), h_cond_interval.size() * sizeof(int8_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_offset_shared_logG, h_offset_shared_logG.data(), h_offset_shared_logG.size() * sizeof(long long), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_expected_energy_spectrum, h_expected_energy_spectrum.data(), h_expected_energy_spectrum.size() * sizeof(int8_t), cudaMemcpyHostToDevice));

    // Launch kernel
    const int max_threads_per_block = 128;
    const int block_count = (total_len_histogram + max_threads_per_block - 1) / max_threads_per_block;

    wl::calc_average_log_g<<<block_count, max_threads_per_block>>>(
        d_shared_logG, d_len_histograms, d_log_G, d_interval_start_energies, d_interval_end_energies, d_expected_energy_spectrum,
        d_cond_interval,  d_offset_histogram, d_offset_energy_spectrum, d_offset_shared_logG, d_cond_interaction,
        num_interactions, num_walker_per_interval, num_intervals_per_interaction, total_len_histogram);

    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy results back to host
    CUDA_CHECK(cudaMemcpy(h_shared_log_G.data(), d_shared_logG, h_shared_log_G.size() * sizeof(double), cudaMemcpyDeviceToHost));

    // Expectation
    std::vector<double> expected_h_shared_log_G(20, 0.0);
    for(std::size_t i = 0; i < total_len_histogram; i++){
        auto iterator = std::lower_bound(h_offset_histogram.begin(), h_offset_histogram.end(), i);
        // index is positive as iterator must be greater equal begin of histogram offsets
        std::size_t index = static_cast<std::size_t>(iterator - h_offset_histogram.begin());
        std::size_t walker_over_all_disorders_id = (iterator != h_offset_histogram.begin() && static_cast<std::size_t>(*iterator) != i) ? index - 1 : index;
        std::size_t interval_over_all_disorders_id = walker_over_all_disorders_id / num_walker_per_interval;
        if (walker_over_all_disorders_id >= h_offset_histogram.size()) {
            throw std::out_of_range("walker_over_all_disorders_id is out of range");
        }
        std::size_t energy_in_histogram =  i - static_cast<std::size_t>(h_offset_histogram[walker_over_all_disorders_id]);
        if(interval_over_all_disorders_id >= num_intervals_per_interaction * num_interactions){
            throw std::out_of_range("interval_over_all_disorders_id is out of range");
        }
        if(h_cond_interval[interval_over_all_disorders_id] == 1){
            expected_h_shared_log_G[static_cast<std::size_t>(h_offset_shared_logG[interval_over_all_disorders_id])+energy_in_histogram] += h_log_G[i]/static_cast<double>(num_walker_per_interval);
        }
    }

    // checks
    for (size_t i = 0; i < h_shared_log_G.size(); ++i) {
        BOOST_CHECK_CLOSE(h_shared_log_G[i], expected_h_shared_log_G[i], 1e-15);
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
    CUDA_CHECK(cudaFree(d_expected_energy_spectrum));
    CUDA_CHECK(cudaFree(d_offset_energy_spectrum));
}

BOOST_AUTO_TEST_CASE(test_calc_average_log_g_not_computed_for_done_interactions)
{
    const int num_interactions = 3;
    const int num_intervals_per_interaction = 3;
    const int num_walker_per_interval = 4;
    const int total_len_histogram = 20*4;

    // Input arrays
    std::vector<double> h_log_G(total_len_histogram, 1.0);
    for(std::size_t i = 0; i < total_len_histogram; i++){
        h_log_G[i] = static_cast<double>(i);
    }
    std::vector<double> h_shared_log_G(20, 0.0);
    std::vector<int> h_len_histograms = {7*4, 9*4, 4*4};
    std::vector<int> h_interval_start_energies = {-2, -1, 0, -4, -2, 0, -2, -1, 0};
    std::vector<int> h_interval_end_energies = {-1, 0, 2, -2, 0, 2, -2, -1, 1};
    std::vector<int> h_cond_interaction = {-1, -1, 1};
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
    std::vector<int8_t> h_cond_interval = {1, 1, 1, 1, 1, 1, 1, 1,1};
    std::vector<long long> h_offset_shared_logG = {0, 2, 4, 7, 10, 13, 16, 17, 18};
    std::vector<int> h_offset_energy_spectrum = {0, 7, 16};
    std::vector<std::int8_t> h_expected_energy_spectrum(20, 1);

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
    int8_t* d_expected_energy_spectrum;
    int* d_offset_energy_spectrum;

    CUDA_CHECK(cudaMalloc(&d_log_G, total_len_histogram * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_shared_logG, h_shared_log_G.size() * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_len_histograms, h_len_histograms.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_interval_start_energies, h_interval_start_energies.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_interval_end_energies, h_interval_end_energies.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cond_interaction, h_cond_interaction.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_offset_histogram, h_offset_histogram.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_offset_energy_spectrum, h_offset_energy_spectrum.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cond_interval, h_cond_interval.size() * sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(&d_offset_shared_logG, h_offset_shared_logG.size() * sizeof(long long)));
    CUDA_CHECK(cudaMalloc(&d_expected_energy_spectrum, h_expected_energy_spectrum.size() * sizeof(std::int8_t)));

    CUDA_CHECK(cudaMemset(d_shared_logG, 0, h_shared_log_G.size() * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(d_log_G, h_log_G.data(), h_log_G.size() * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_len_histograms, h_len_histograms.data(), h_len_histograms.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_interval_start_energies, h_interval_start_energies.data(), h_interval_start_energies.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_interval_end_energies, h_interval_end_energies.data(), h_interval_end_energies.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cond_interaction, h_cond_interaction.data(), h_cond_interaction.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_offset_histogram, h_offset_histogram.data(), h_offset_histogram.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_offset_energy_spectrum, h_offset_energy_spectrum.data(), h_offset_energy_spectrum.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cond_interval, h_cond_interval.data(), h_cond_interval.size() * sizeof(int8_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_offset_shared_logG, h_offset_shared_logG.data(), h_offset_shared_logG.size() * sizeof(long long), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_expected_energy_spectrum, h_expected_energy_spectrum.data(), h_expected_energy_spectrum.size() * sizeof(int8_t), cudaMemcpyHostToDevice));

    // Launch kernel
    const int max_threads_per_block = 128;
    const int block_count = (total_len_histogram + max_threads_per_block - 1) / max_threads_per_block;

    wl::calc_average_log_g<<<block_count, max_threads_per_block>>>(
        d_shared_logG, d_len_histograms, d_log_G, d_interval_start_energies, d_interval_end_energies, d_expected_energy_spectrum,
        d_cond_interval,  d_offset_histogram, d_offset_energy_spectrum, d_offset_shared_logG, d_cond_interaction,
        num_interactions, num_walker_per_interval, num_intervals_per_interaction, total_len_histogram);

    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy results back to host
    CUDA_CHECK(cudaMemcpy(h_shared_log_G.data(), d_shared_logG, h_shared_log_G.size() * sizeof(double), cudaMemcpyDeviceToHost));

    // Expectation
    std::vector<double> expected_h_shared_log_G(20, 0.0);
    for(std::size_t i = 0; i < total_len_histogram; i++){
        auto iterator = std::lower_bound(h_offset_histogram.begin(), h_offset_histogram.end(), i);
        // index is positive as iterator must be greater equal begin of histogram offsets
        std::size_t index = static_cast<std::size_t>(iterator - h_offset_histogram.begin());
        std::size_t walker_over_all_disorders_id = (iterator != h_offset_histogram.begin() && static_cast<std::size_t>(*iterator) != i) ? index - 1 : index;
        std::size_t interval_over_all_disorders_id = walker_over_all_disorders_id / num_walker_per_interval;
        if (walker_over_all_disorders_id >= h_offset_histogram.size()) {
            throw std::out_of_range("walker_over_all_disorders_id is out of range");
        }
        std::size_t energy_in_histogram =  i - static_cast<std::size_t>(h_offset_histogram[walker_over_all_disorders_id]);
        if(interval_over_all_disorders_id >= num_intervals_per_interaction * num_interactions){
            throw std::out_of_range("interval_over_all_disorders_id is out of range");
        }
        if(h_cond_interval[interval_over_all_disorders_id] == 1 && h_cond_interaction[walker_over_all_disorders_id/(num_walker_per_interval*num_intervals_per_interaction)]==1){
            expected_h_shared_log_G[static_cast<std::size_t>(h_offset_shared_logG[interval_over_all_disorders_id])+energy_in_histogram] += h_log_G[i]/static_cast<double>(num_walker_per_interval);
        }
    }

    // checks
    for (size_t i = 0; i < h_shared_log_G.size(); ++i) {
        BOOST_CHECK_CLOSE(h_shared_log_G[i], expected_h_shared_log_G[i], 1e-15);
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
    CUDA_CHECK(cudaFree(d_expected_energy_spectrum));
    CUDA_CHECK(cudaFree(d_offset_energy_spectrum));
}

BOOST_AUTO_TEST_SUITE_END()