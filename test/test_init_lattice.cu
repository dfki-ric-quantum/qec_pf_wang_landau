#define BOOST_TEST_MODULE wl_tests

#include <cstdint>
#include <random>
#include <vector>

#include <boost/test/tools/old/interface.hpp>
#include <boost/test/unit_test.hpp>
#include <cuda_runtime.h>

#include "../src/cuda_utils.cuh"
#include "../src/wl.cuh"

BOOST_AUTO_TEST_SUITE(init_lattice)

/**
 * @brief This test checks whether the wanted percentage of spins is set to -1
 * within a tolerance of 0.02. It further checks whether all spins are either +1
 * or -1.
 */
BOOST_AUTO_TEST_CASE(check_consistent_percentages)
{
  std::random_device random_device;
  const int seed = static_cast<int>(random_device());

  const int dimX = 100;
  const int dimY = 100;

  const int num_lattices = 10;
  const int walker_per_interactions = 5;

  const int THREADS = 128;
  const int BLOCKS = (num_lattices * dimY * dimX * 2 + THREADS - 1) / THREADS;

  wl::device_ptr<std::int8_t> d_lattice(dimX * dimY * num_lattices, 0);
  wl::device_ptr<float> d_init_probs_per_lattice(num_lattices, 0);

  wl::init_lattice<<<BLOCKS, THREADS>>>(
    d_lattice.data(), d_init_probs_per_lattice.data(), dimX, dimY, num_lattices, seed, walker_per_interactions);

  std::vector<std::int8_t> h_lattice(dimX * dimY * num_lattices, 0);
  std::vector<float> h_init_probs_per_lattice(num_lattices, 0);

  d_lattice.device_to_host(h_lattice);
  d_init_probs_per_lattice.device_to_host(h_init_probs_per_lattice);

  std::vector<float> percentage_flips_per_lattice(num_lattices, 0);

  for (int i = 0; i < num_lattices; i++) {
    int count = 0;

    for (int x = 0; x < dimX; x++) {
      for (int y = 0; y < dimY; y++) {
        std::int8_t spin = h_lattice[static_cast<std::size_t>(i * dimX * dimY + x * dimY + y)];

        BOOST_CHECK(spin == 1 || spin == -1);

        if (spin == -1) {
          count += 1;
        }
      }
    }

    percentage_flips_per_lattice[static_cast<std::size_t>(i)] = static_cast<float>(count) / (dimX * dimY);
  }

  const float tolerance = 0.02F;

  for (std::size_t i = 0; i < num_lattices; i++) {
    BOOST_CHECK(std::abs(percentage_flips_per_lattice[i] - h_init_probs_per_lattice[i]) < tolerance);
  }
}

BOOST_AUTO_TEST_SUITE_END()
