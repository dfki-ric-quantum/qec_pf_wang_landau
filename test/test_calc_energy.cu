#define BOOST_TEST_MODULE wl_tests

#include <cstdint>
#include <vector>

#include <cuda_runtime.h>
#include <boost/test/tools/old/interface.hpp>
#include <boost/test/unit_test.hpp>

#include "../src/cuda_utils.cuh"
#include "../src/wl.cuh"


BOOST_AUTO_TEST_SUITE(calc_energy)

/**
 * @brief In the all_up testcase, we initialize 10 lattices divided over two interactions.
 * All spins and interactions are initialized to +1, such that the energy should be equal
 * to -2*dimX*dimY.
 */

BOOST_AUTO_TEST_CASE(all_up_periodic)
{
    const int dimX = 4;
    const int dimY = 4;

    const int num_lattices = 10;
    const int num_interactions = 2;
    const int walker_per_interactions = 5;

    const int THREADS = 128;
    const int BLOCKS = (num_lattices + THREADS - 1) / THREADS;

    std::vector<std::int8_t> h_lattice(dimX*dimY*num_lattices, 1);
    std::vector<std::int8_t> h_interactions(2*dimX*dimY*num_interactions, 1);

    std::vector<int> h_offset_lattice(num_lattices, 0);

    for (int i = 0; i < num_lattices; i++){
        h_offset_lattice[i] = i*dimX*dimY;
    }

    std::vector<int> h_energy(num_lattices, 0);

    wl::device_ptr<int> d_energy(num_lattices, 0);

    wl::device_ptr<std::int8_t> d_lattice(dimX*dimY*num_lattices, 0);
    wl::device_ptr<std::int8_t> d_interactions(2*dimX*dimY*num_interactions, 0);
    wl::device_ptr<int> d_offset_lattice(num_lattices, 0);

    d_lattice.host_to_device(h_lattice);
    d_interactions.host_to_device(h_interactions);
    d_offset_lattice.host_to_device(h_offset_lattice);

    wl::calc_energy_periodic_boundary<<<BLOCKS, THREADS>>>(d_energy.data(), d_lattice.data(), d_interactions.data(),
                                                           d_offset_lattice.data(), dimX, dimY, num_lattices, walker_per_interactions);

    d_energy.device_to_host(h_energy);

    std::vector<int> real_energies(num_lattices, dimX*dimY*(-2));

    BOOST_CHECK_EQUAL_COLLECTIONS(h_energy.begin(), h_energy.end(), real_energies.begin(), real_energies.end());
}


BOOST_AUTO_TEST_CASE(all_up_open)
{
    const int dimX = 4;
    const int dimY = 4;

    const int num_lattices = 10;
    const int num_interactions = 2;
    const int walker_per_interactions = 5;

    const int THREADS = 128;
    const int BLOCKS = (num_lattices + THREADS - 1) / THREADS;

    std::vector<std::int8_t> h_lattice(dimX*dimY*num_lattices, 1);
    std::vector<std::int8_t> h_interactions(2*dimX*dimY*num_interactions, 1);

    std::vector<int> h_offset_lattice(num_lattices, 0);

    for (int i = 0; i < num_lattices; i++){
        h_offset_lattice[i] = i*dimX*dimY;
    }

    std::vector<int> h_energy(num_lattices, 0);

    wl::device_ptr<int> d_energy(num_lattices, 0);

    wl::device_ptr<std::int8_t> d_lattice(dimX*dimY*num_lattices, 0);
    wl::device_ptr<std::int8_t> d_interactions(2*dimX*dimY*num_interactions, 0);
    wl::device_ptr<int> d_offset_lattice(num_lattices, 0);

    d_lattice.host_to_device(h_lattice);
    d_interactions.host_to_device(h_interactions);
    d_offset_lattice.host_to_device(h_offset_lattice);

    wl::calc_energy_open_boundary<<<BLOCKS, THREADS>>>(d_energy.data(), d_lattice.data(), d_interactions.data(),
                                                           d_offset_lattice.data(), dimX, dimY, num_lattices, walker_per_interactions);

    d_energy.device_to_host(h_energy);

    std::vector<int> real_energies(num_lattices, dimX*dimY*(-2)+dimX+dimY);

    BOOST_CHECK_EQUAL_COLLECTIONS(h_energy.begin(), h_energy.end(), real_energies.begin(), real_energies.end());
}

/**
 * @brief In this test case we initialize one lattice and one interaction with a known energy
 * and compare it to the calculated energy.
 *
 */

BOOST_AUTO_TEST_CASE(single_lattice_periodic)
{
    const int dimX = 4;
    const int dimY = 4;

    const int num_lattices = 1;
    const int num_interactions = 1;
    const int walker_per_interactions = 1;

    const int THREADS = 128;
    const int BLOCKS = (num_lattices + THREADS - 1) / THREADS;

    std::vector<std::int8_t> h_lattice = {1, 1,-1,1,
                                          -1, -1, 1,-1,
                                          1,-1,1,-1,
                                          -1, -1, -1, -1};

    std::vector<std::int8_t> h_interactions = {1,-1,1,1,
                                               1,1,1,1,
                                               -1,1,1,-1,
                                               1,1,1,1,
                                               1,1,-1,1,
                                               1,1,1,1,
                                               -1,1,1,1,
                                               1,1,1,1};

    std::vector<int> h_offset_lattice(num_lattices, 0);
    std::vector<int> h_energy(num_lattices, 0);

    wl::device_ptr<int> d_energy(num_lattices, 0);
    wl::device_ptr<std::int8_t> d_lattice(dimX*dimY*num_lattices, 0);
    wl::device_ptr<std::int8_t> d_interactions(2*dimX*dimY*num_interactions, 0);
    wl::device_ptr<int> d_offset_lattice(num_lattices, 0);

    d_lattice.host_to_device(h_lattice);
    d_interactions.host_to_device(h_interactions);
    d_offset_lattice.host_to_device(h_offset_lattice);

    wl::calc_energy_periodic_boundary<<<BLOCKS, THREADS>>>(d_energy.data(), d_lattice.data(), d_interactions.data(),
                                                           d_offset_lattice.data(), dimX, dimY, num_lattices, walker_per_interactions);

    d_energy.device_to_host(h_energy);

    BOOST_CHECK(h_energy[0] == -6);
}

/**
 * @brief In this test case we initialize one lattice and one interaction with a known energy
 * and compare it to the calculated energy.
 *
 */

BOOST_AUTO_TEST_CASE(single_lattice_open)
{
    const int dimX = 4;
    const int dimY = 4;

    const int num_lattices = 1;
    const int num_interactions = 1;
    const int walker_per_interactions = 1;

    const int THREADS = 128;
    const int BLOCKS = (num_lattices + THREADS - 1) / THREADS;

    std::vector<std::int8_t> h_lattice = {1, 1,-1,1,
                                          -1, -1, 1,-1,
                                          1,-1,1,-1,
                                          -1, -1, -1, -1};

    std::vector<std::int8_t> h_interactions = {1,-1,1,1,
                                               1,1,1,1,
                                               -1,1,1,-1,
                                               1,1,1,1,
                                               1,1,-1,1,
                                               1,1,1,1,
                                               -1,1,1,1,
                                               1,1,1,1};

    std::vector<int> h_offset_lattice(num_lattices, 0);
    std::vector<int> h_energy(num_lattices, 0);

    wl::device_ptr<int> d_energy(num_lattices, 0);
    wl::device_ptr<std::int8_t> d_lattice(dimX*dimY*num_lattices, 0);
    wl::device_ptr<std::int8_t> d_interactions(2*dimX*dimY*num_interactions, 0);
    wl::device_ptr<int> d_offset_lattice(num_lattices, 0);

    d_lattice.host_to_device(h_lattice);
    d_interactions.host_to_device(h_interactions);
    d_offset_lattice.host_to_device(h_offset_lattice);

    wl::calc_energy_open_boundary<<<BLOCKS, THREADS>>>(d_energy.data(), d_lattice.data(), d_interactions.data(),
                                                           d_offset_lattice.data(), dimX, dimY, num_lattices, walker_per_interactions);

    d_energy.device_to_host(h_energy);

    BOOST_CHECK(h_energy[0] == -4);
}

BOOST_AUTO_TEST_SUITE_END()
