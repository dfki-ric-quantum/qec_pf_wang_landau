#include <boost/test/tools/old/interface.hpp>
#define BOOST_TEST_MODULE wl_tests

#include <vector>
#include <random>
#include <thrust/reduce.h>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>

#include <cuda_runtime.h>
#include <boost/test/unit_test.hpp>

#include "../src/interval.hpp"
#include "../src/cuda_utils.cuh"
#include "../src/wl.cuh"


BOOST_AUTO_TEST_SUITE(check_energy_ranges)

/** 
In this test we first initialize the energy array with correct energies fitting the intervals. After that we add some energies which are
outside of the energy ranges to see whether the check can filter them out.*/

BOOST_AUTO_TEST_CASE(check_correct_ranges)
{
    const int dimX = 10;
    const int dimY = 10;

    const int num_intervals = 15;
    const int num_walker_per_interval = 5;
    const int num_interactions = 100;

    const int total_intervals = num_interactions * num_intervals;
    const int num_total_walker = num_interactions*num_intervals*num_walker_per_interval;

    const float overlap = 0.25;

    std::vector<int> Emin(num_interactions, -2*dimX*dimY);
    std::vector<int> Emax(num_interactions, 2*dimX*dimY);

    std::vector<int> h_start_interval_energies;
    std::vector<int> h_end_interval_energies;

    for (int i=0; i < num_interactions; i++){
        wl::interval_results result{Emin[i], Emax[i], num_intervals, num_walker_per_interval, overlap};

        h_start_interval_energies.insert(h_start_interval_energies.end(), result.h_start.begin(), result.h_start.end());
        h_end_interval_energies.insert(h_end_interval_energies.end(), result.h_end.begin(), result.h_end.end());
    }

    // Initialize energy per walker
    std::vector<int> h_energy_per_walker(num_total_walker, 0);

    std::random_device rd;  // Seed
    std::mt19937 gen(rd()); // Mersenne Twister engine

    for (int i = 0; i < num_interactions; i++){
        for (int j=0; j < num_intervals; j++){
            std::uniform_int_distribution<> distr(h_start_interval_energies[i*num_intervals +j], h_end_interval_energies[i*num_intervals +j]);
            for (int k = 0; k < num_walker_per_interval; k++){
                int offset = i*num_intervals*num_walker_per_interval + j*num_walker_per_interval + k;
                
                h_energy_per_walker[offset] = distr(gen);
            }
        }
    }
    
    wl::device_ptr<int> d_energy_per_walker(h_energy_per_walker);
    wl::device_ptr<int> d_start_interval_energies(h_start_interval_energies);
    wl::device_ptr<int> d_end_interval_energies(h_end_interval_energies);
    wl::device_ptr<int8_t> d_flag_check_energies(num_total_walker, 0);

    wl::check_energy_ranges<<<total_intervals, num_walker_per_interval>>>(d_flag_check_energies.data(), 
                                                                          d_energy_per_walker.data(),
                                                                          d_start_interval_energies.data(),
                                                                          d_end_interval_energies.data(),
                                                                          num_total_walker);

    int max_energy_check = 0;

    thrust::device_ptr<int8_t> d_check_energy_pointer(d_flag_check_energies.data());
    thrust::device_ptr<int8_t> max_flag_check_energies_ptr = thrust::max_element(d_check_energy_pointer, d_check_energy_pointer + num_total_walker);
    max_energy_check = *max_flag_check_energies_ptr;

    BOOST_CHECK(max_energy_check == 0);

    if (max_energy_check == 0){
        printf("First check with correct energies correct \n");
    }

    // Set some wrong energies to test behavior
    std::uniform_real_distribution<> dist(0, 1);
    
    for (int i = 0; i < num_interactions; i++){
        for (int j=0; j < num_intervals; j++){
            std::uniform_int_distribution<> distr(h_start_interval_energies[i*num_intervals +j], h_end_interval_energies[i*num_intervals +j]);
            for (int k = 0; k < num_walker_per_interval; k++){
                int offset = i*num_intervals*num_walker_per_interval + j*num_walker_per_interval + k;
                
                double random_uniform = dist(gen);

                if (random_uniform < 0.05){
                    h_energy_per_walker[offset] = distr(gen);
                }
                else{
                    h_energy_per_walker[offset] = Emax[0] + 10;
                }
            }
        }
    }

    d_energy_per_walker.host_to_device(h_energy_per_walker);

    wl::check_energy_ranges<<<total_intervals, num_walker_per_interval>>>(d_flag_check_energies.data(), 
                                                                        d_energy_per_walker.data(),
                                                                        d_start_interval_energies.data(),
                                                                        d_end_interval_energies.data(),
                                                                        num_total_walker);

    int max_energy_fail = 0;

    thrust::device_ptr<int8_t> d_fail_energy_pointer(d_flag_check_energies.data());
    thrust::device_ptr<int8_t> max_fail_check_energies_ptr = thrust::max_element(d_fail_energy_pointer, d_fail_energy_pointer + num_total_walker);
    max_energy_fail = *max_fail_check_energies_ptr;

    BOOST_CHECK(max_energy_fail == 1);
}

BOOST_AUTO_TEST_SUITE_END()
