#include <boost/test/tools/old/interface.hpp>
#include <cstdint>
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


BOOST_AUTO_TEST_SUITE(check_histogram)

BOOST_AUTO_TEST_CASE(test_check_histogram_success)
{
    const double alpha = 0.8;
    const int num_interactions = 3;
    const int num_intervals_per_interaction = 3;
    const int num_walker_per_interval = 4;
    const int total_len_histogram = 20*4;

    // Input arrays
    std::vector<unsigned long long> h_histogram(total_len_histogram, 10);
    h_histogram[0] = 1;  // first walker in first interval of first interaction ist not flat and thus shall prohibit reset of histogram

    std::vector<int> h_len_histograms = {7*4, 9*4, 4*4};
    std::vector<int> h_interval_start_energies = {-2, -1, 0, -4, -2, 0, -2, -1, 0};
    std::vector<int> h_interval_end_energies = {-1, 0, 2, -2, 0, 2, -2, -1, 1};
    std::vector<int> h_cond_interaction = {1, 1, 1}; // all interactions arent done and have to be checked in kernel
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
    std::vector<int8_t> h_cond_interval = {0, 0, 0, 0, 0, 0, 0, 0, 1}; // the 1 at the end shall prohibit reset of the walker histograms of the last interval
    std::vector<int> h_offset_energy_spectrum = {0, 7, 16};
    std::vector<std::int8_t> h_expected_energy_spectrum(20, 1);
    std::vector<double> h_factor(num_interactions * num_walker_per_interval * num_intervals_per_interaction, std::numbers::e);

    // Device arrays
    unsigned long long* d_histogram;
    double *d_factor;
    int* d_len_histograms;
    int* d_interval_start_energies;
    int* d_interval_end_energies;
    int* d_cond_interaction;
    int* d_offset_histogram;
    int8_t* d_cond_interval;
    int8_t* d_expected_energy_spectrum;
    int* d_offset_energy_spectrum;

    CUDA_CHECK(cudaMalloc(&d_histogram, total_len_histogram * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_factor, h_factor.size() * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_len_histograms, h_len_histograms.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_interval_start_energies, h_interval_start_energies.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_interval_end_energies, h_interval_end_energies.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cond_interaction, h_cond_interaction.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_offset_histogram, h_offset_histogram.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_offset_energy_spectrum, h_offset_energy_spectrum.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cond_interval, h_cond_interval.size() * sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(&d_expected_energy_spectrum, h_expected_energy_spectrum.size() * sizeof(std::int8_t)));


    CUDA_CHECK(cudaMemcpy(d_histogram, h_histogram.data(), h_histogram.size() * sizeof(unsigned long long), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_factor, h_factor.data(), h_factor.size() * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_len_histograms, h_len_histograms.data(), h_len_histograms.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_interval_start_energies, h_interval_start_energies.data(), h_interval_start_energies.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_interval_end_energies, h_interval_end_energies.data(), h_interval_end_energies.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cond_interaction, h_cond_interaction.data(), h_cond_interaction.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_offset_histogram, h_offset_histogram.data(), h_offset_histogram.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_offset_energy_spectrum, h_offset_energy_spectrum.data(), h_offset_energy_spectrum.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cond_interval, h_cond_interval.data(), h_cond_interval.size() * sizeof(int8_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_expected_energy_spectrum, h_expected_energy_spectrum.data(), h_expected_energy_spectrum.size() * sizeof(int8_t), cudaMemcpyHostToDevice));

    // Launch kernel
    const unsigned int total_intervals = num_interactions * num_intervals_per_interaction;
    wl::check_histogram<<<total_intervals, num_walker_per_interval>>>(d_histogram,
                                                                    d_factor,
                                                                    d_cond_interval,
                                                                    d_offset_histogram,
                                                                    d_interval_end_energies,
                                                                    d_interval_start_energies,
                                                                    d_expected_energy_spectrum,
                                                                    d_offset_energy_spectrum,
                                                                    d_cond_interaction,
                                                                    alpha,
                                                                    num_walker_per_interval*num_intervals_per_interaction*num_interactions,
                                                                    num_walker_per_interval*num_intervals_per_interaction,
                                                                    num_intervals_per_interaction);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy results back to host
    CUDA_CHECK(cudaMemcpy(h_histogram.data(), d_histogram,  total_len_histogram* sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_cond_interval.data(), d_cond_interval,  h_cond_interval.size()* sizeof(int8_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_factor.data(), d_factor,  h_factor.size()* sizeof(double), cudaMemcpyDeviceToHost));

    // Expectations
    std::vector<unsigned long long> h_histogram_expected(total_len_histogram, 0);
    for(std::size_t i = 0; i<h_histogram_expected.size(); i++){
        if(i<8 || i>71){
            h_histogram_expected[i]=10;
        }
    }
    h_histogram_expected[0]=1;
    std::vector<int8_t> h_cond_interval_expected = {0, 1, 1,1,1,1,1,1,1};
    std::vector<double> h_factor_expected(num_interactions * num_walker_per_interval * num_intervals_per_interaction, sqrt(std::numbers::e));
    for(std::size_t i = 0; i<h_factor_expected.size(); i++){
        if(i/num_walker_per_interval==0 || i/num_walker_per_interval == 8){
            h_factor_expected[i]=std::numbers::e;
        }
    }

    // Checks
    for (size_t i = 0; i < h_histogram_expected.size(); ++i) {
        BOOST_CHECK_EQUAL(h_histogram_expected[i], h_histogram[i]);
    }
    for (size_t i = 0; i < h_cond_interval_expected.size(); ++i) {
        BOOST_CHECK_EQUAL(h_cond_interval_expected[i], h_cond_interval[i]);
    }
    for (size_t i = 0; i < h_factor_expected.size(); ++i) {
        BOOST_CHECK_CLOSE(h_factor_expected[i], h_factor[i], 1e-15);
    }

    // Free device memory
    CUDA_CHECK(cudaFree(d_histogram));
    CUDA_CHECK(cudaFree(d_len_histograms));
    CUDA_CHECK(cudaFree(d_interval_start_energies));
    CUDA_CHECK(cudaFree(d_interval_end_energies));
    CUDA_CHECK(cudaFree(d_cond_interaction));
    CUDA_CHECK(cudaFree(d_offset_histogram));
    CUDA_CHECK(cudaFree(d_cond_interval));
    CUDA_CHECK(cudaFree(d_expected_energy_spectrum));
    CUDA_CHECK(cudaFree(d_offset_energy_spectrum));
    CUDA_CHECK(cudaFree(d_factor));
}


BOOST_AUTO_TEST_SUITE_END()