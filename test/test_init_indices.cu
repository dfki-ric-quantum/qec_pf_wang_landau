#include <boost/test/tools/old/interface.hpp>
#define BOOST_TEST_MODULE wl_tests

#include <algorithm>
#include <cstdint>
#include <vector>

#include <cuda_runtime.h>
#include <boost/test/unit_test.hpp>

#include "../src/cuda_utils.cuh"
#include "../src/wl.cuh"

BOOST_AUTO_TEST_SUITE(init_indices)

BOOST_AUTO_TEST_CASE(test_indices)
{
    const int walker_per_interval = 10;
    const int num_interactions = 100;
    const int num_intervals_per_interaction = 10;
    
    const int total_intervals = num_interactions * num_intervals_per_interaction;
    const int total_walker = num_interactions * num_intervals_per_interaction * walker_per_interval;

    wl::device_ptr<int> d_replica_indices(total_walker, 0);

    wl::init_indices<<<total_intervals, walker_per_interval>>>(d_replica_indices.data(), total_walker);

    std::vector<int> h_replica_indices(total_walker, 0);
    d_replica_indices.device_to_host(h_replica_indices);

    for (int i = 0; i < total_intervals; i++){
        for (int j = 0; j < walker_per_interval; j++){
            BOOST_CHECK(h_replica_indices[i*walker_per_interval + j] == j);
        }
    }
}

BOOST_AUTO_TEST_SUITE_END()