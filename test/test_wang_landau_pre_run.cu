#define BOOST_TEST_MODULE wl_tests

#include <cstdint>
#include <ctime>
#include <random>
#include <vector>

#include <boost/test/tools/old/interface.hpp>
#include <boost/test/unit_test.hpp>
#include <cuda_runtime.h>

#include "../src/cuda_utils.cuh"
#include "../src/interval.hpp"
#include "../src/wl.cuh"

BOOST_AUTO_TEST_SUITE(wang_landau_pre_run)

/**
 * @brief In the single iteration periodic test we test two things. The first thing is whether we flip exactly one spin per lattice
 * for a single Wang Landau iteration. The second test is, whether the energy after the flip spin calculated through
 * one wang_landau iteration is consistent with the new spin configuration.
 *
 */
BOOST_AUTO_TEST_CASE(single_iteration_periodic)
{
  const int dimX = 12;
  const int dimY = 12;
  const std::size_t lattice_size = dimX * dimY;

  const int num_interactions = 3;
  const int walker_per_interactions = 10;
  const int num_lattices = num_interactions * walker_per_interactions;

  const int E_min = -2 * dimX * dimY;
  const int E_max = -E_min;

  const int len_single_histogram = E_max - E_min + 1;
  const int len_total_histogram = num_interactions * len_single_histogram;

  const int num_intervals = 5;

  const int num_wl_iterations = 1;

  const int THREADS = 128;
  const int BLOCKS = (num_lattices * dimY * dimX * 2 + THREADS - 1) / THREADS;
  const int BLOCKS_LATTICE = (num_lattices + THREADS - 1) / THREADS;

  std::random_device rd;
  const int seed = static_cast<int>(rd());

  std::default_random_engine generator(seed);

  std::uniform_real_distribution<float> distribution(0.0, 1.0);
  float prob = distribution(generator);

  wl::interval_results result{E_min, E_max, num_intervals, 1, 1.0F};

  // Lattice initialization
  wl::device_ptr<std::int8_t> d_lattice(dimX * dimY * num_lattices, 0);
  wl::device_ptr<float> d_init_probs_per_lattice(num_lattices, 0);

  wl::init_lattice<<<BLOCKS, THREADS>>>(
    d_lattice.data(), d_init_probs_per_lattice.data(), dimX, dimY, num_lattices, seed, walker_per_interactions);

  // Interaction initialization
  wl::device_ptr<std::int8_t> d_interactions(num_interactions * dimX * dimY * 2, 0);
  wl::init_interactions<<<BLOCKS, THREADS>>>(d_interactions.data(), lattice_size, num_interactions, seed + 1, prob);

  // Lattice Offsets
  wl::device_ptr<int> d_offset_lattice(num_lattices, 0);
  wl::init_offsets_lattice<<<BLOCKS_LATTICE, THREADS>>>(d_offset_lattice.data(), dimX, dimY, num_lattices);

  // Places to store intervals
  wl::device_ptr<std::int8_t> d_found_interval_lattice(dimX * dimY * num_intervals * num_interactions, 0);
  wl::device_ptr<int> d_flag_found_interval(num_intervals * num_interactions, 0);

  // Histogram initialization
  wl::device_ptr<unsigned long long> d_histogram(len_total_histogram, 0);

  // Seed offset initialization
  wl::device_ptr<unsigned long> d_seed_offset(num_lattices, 0);

  // Energy initialization
  wl::device_ptr<int> d_energy(num_lattices, 0);
  wl::device_ptr<int> d_test_energy(num_lattices, 0);

  // Host Lattice
  std::vector<std::int8_t> h_lattice_before(num_lattices * dimX * dimY, 0);
  std::vector<std::int8_t> h_lattice_after(num_lattices * dimX * dimY, 0);

  d_lattice.device_to_host(h_lattice_before);

  wl::calc_energy_periodic_boundary<<<BLOCKS_LATTICE, THREADS>>>(d_energy.data(),
                                                                 d_lattice.data(),
                                                                 d_interactions.data(),
                                                                 d_offset_lattice.data(),
                                                                 dimX,
                                                                 dimY,
                                                                 num_lattices,
                                                                 walker_per_interactions);

  wl::wang_landau_pre_run<<<BLOCKS_LATTICE, THREADS>>>(d_energy.data(),
                                                       d_flag_found_interval.data(),
                                                       d_lattice.data(),
                                                       d_found_interval_lattice.data(),
                                                       d_histogram.data(),
                                                       d_seed_offset.data(),
                                                       d_interactions.data(),
                                                       d_offset_lattice.data(),
                                                       E_min,
                                                       E_max,
                                                       num_wl_iterations,
                                                       dimX,
                                                       dimY,
                                                       seed,
                                                       num_intervals,
                                                       result.len_interval,
                                                       num_lattices,
                                                       walker_per_interactions,
                                                       wl::boundary::periodic);

  d_lattice.device_to_host(h_lattice_after);

  wl::calc_energy_periodic_boundary<<<BLOCKS_LATTICE, THREADS>>>(d_test_energy.data(),
                                                                 d_lattice.data(),
                                                                 d_interactions.data(),
                                                                 d_offset_lattice.data(),
                                                                 dimX,
                                                                 dimY,
                                                                 num_lattices,
                                                                 walker_per_interactions);

  // Copy energies and compare
  std::vector<int> h_energy(num_lattices, 0);
  std::vector<int> h_test_energy(num_lattices, 0);

  d_energy.device_to_host(h_energy);
  d_test_energy.device_to_host(h_test_energy);

  // CHeck if energies are the same
  for (int i = 0; i < num_lattices; i++) {
    BOOST_CHECK(h_energy[i] == h_test_energy[i]);
  }

  // Check if only one spin at most is flipped
  for (int i = 0; i < num_lattices; i++) {
    int count_flips = 0;

    for (int j = 0; j < dimX * dimY; j++) {
      if (h_lattice_before[i * dimX * dimY + j] != h_lattice_after[i * dimX * dimY + j]) {
        count_flips += 1;
      }
    }

    BOOST_CHECK(count_flips == 1);
  }
}

/**
 * @brief In the single iteration open test we test two things. The first thing is whether we flip exactly one spin per lattice
 * for a single Wang Landau iteration. The second test is, whether the energy after the flip spin calculated through
 * one wang_landau iteration is consistent with the new spin configuration.
 *
 */
BOOST_AUTO_TEST_CASE(single_iteration_open)
{
  const int dimX = 12;
  const int dimY = 12;
  const std::size_t lattice_size = dimX * dimY;

  const int num_interactions = 3;
  const int walker_per_interactions = 10;
  const int num_lattices = num_interactions * walker_per_interactions;

  const int E_min = -2 * dimX * dimY;
  const int E_max = -E_min;

  const int len_single_histogram = E_max - E_min + 1;
  const int len_total_histogram = num_interactions * len_single_histogram;

  const int num_intervals = 5;

  const int num_wl_iterations = 1;

  const int THREADS = 128;
  const int BLOCKS = (num_lattices * dimY * dimX * 2 + THREADS - 1) / THREADS;
  const int BLOCKS_LATTICE = (num_lattices + THREADS - 1) / THREADS;

  std::random_device rd;
  const int seed = static_cast<int>(rd());

  std::default_random_engine generator(seed);

  std::uniform_real_distribution<float> distribution(0.0, 1.0);
  float prob = distribution(generator);

  wl::interval_results result{E_min, E_max, num_intervals, 1, 1.0F};

  // Lattice initialization
  wl::device_ptr<std::int8_t> d_lattice(dimX * dimY * num_lattices, 0);
  wl::device_ptr<float> d_init_probs_per_lattice(num_lattices, 0);

  wl::init_lattice<<<BLOCKS, THREADS>>>(
    d_lattice.data(), d_init_probs_per_lattice.data(), dimX, dimY, num_lattices, seed, walker_per_interactions);

  // Interaction initialization
  wl::device_ptr<std::int8_t> d_interactions(num_interactions * dimX * dimY * 2, 0);
  wl::init_interactions<<<BLOCKS, THREADS>>>(d_interactions.data(), lattice_size, num_interactions, seed + 1, prob);

  // Lattice Offsets
  wl::device_ptr<int> d_offset_lattice(num_lattices, 0);
  wl::init_offsets_lattice<<<BLOCKS_LATTICE, THREADS>>>(d_offset_lattice.data(), dimX, dimY, num_lattices);

  // Places to store intervals
  wl::device_ptr<std::int8_t> d_found_interval_lattice(dimX * dimY * num_intervals * num_interactions, 0);
  wl::device_ptr<int> d_flag_found_interval(num_intervals * num_interactions, 0);

  // Histogram initialization
  wl::device_ptr<unsigned long long> d_histogram(len_total_histogram, 0);

  // Seed offset initialization
  wl::device_ptr<unsigned long> d_seed_offset(num_lattices, 0);

  // Energy initialization
  wl::device_ptr<int> d_energy(num_lattices, 0);
  wl::device_ptr<int> d_test_energy(num_lattices, 0);

  // Host Lattice
  std::vector<std::int8_t> h_lattice_before(num_lattices * dimX * dimY, 0);
  std::vector<std::int8_t> h_lattice_after(num_lattices * dimX * dimY, 0);

  d_lattice.device_to_host(h_lattice_before);

  wl::calc_energy_open_boundary<<<BLOCKS_LATTICE, THREADS>>>(d_energy.data(),
                                                                 d_lattice.data(),
                                                                 d_interactions.data(),
                                                                 d_offset_lattice.data(),
                                                                 dimX,
                                                                 dimY,
                                                                 num_lattices,
                                                                 walker_per_interactions);

  wl::wang_landau_pre_run<<<BLOCKS_LATTICE, THREADS>>>(d_energy.data(),
                                                       d_flag_found_interval.data(),
                                                       d_lattice.data(),
                                                       d_found_interval_lattice.data(),
                                                       d_histogram.data(),
                                                       d_seed_offset.data(),
                                                       d_interactions.data(),
                                                       d_offset_lattice.data(),
                                                       E_min,
                                                       E_max,
                                                       num_wl_iterations,
                                                       dimX,
                                                       dimY,
                                                       seed,
                                                       num_intervals,
                                                       result.len_interval,
                                                       num_lattices,
                                                       walker_per_interactions,
                                                       wl::boundary::open);

  d_lattice.device_to_host(h_lattice_after);

  wl::calc_energy_open_boundary<<<BLOCKS_LATTICE, THREADS>>>(d_test_energy.data(),
                                                                 d_lattice.data(),
                                                                 d_interactions.data(),
                                                                 d_offset_lattice.data(),
                                                                 dimX,
                                                                 dimY,
                                                                 num_lattices,
                                                                 walker_per_interactions);

  // Copy energies and compare
  std::vector<int> h_energy(num_lattices, 0);
  std::vector<int> h_test_energy(num_lattices, 0);

  d_energy.device_to_host(h_energy);
  d_test_energy.device_to_host(h_test_energy);

  // CHeck if energies are the same
  for (int i = 0; i < num_lattices; i++) {
    BOOST_CHECK(h_energy[i] == h_test_energy[i]);
  }

  // Check if only one spin at most is flipped
  for (int i = 0; i < num_lattices; i++) {
    int count_flips = 0;

    for (int j = 0; j < dimX * dimY; j++) {
      if (h_lattice_before[i * dimX * dimY + j] != h_lattice_after[i * dimX * dimY + j]) {
        count_flips += 1;
      }
    }

    BOOST_CHECK(count_flips == 1);
  }
}

/**
 * @brief In this test we test a loop of Wang Landau calls and check if the energies calculated from the Wang Landau
 * kernel match with the energies calculated from the periodic calc energy kernel. It further checks whether the found
 * intervals do fit into the intervals where they should belong to.
 *
 */
BOOST_AUTO_TEST_CASE(multiple_iterations_periodic)
{
  const int dimX = 6;
  const int dimY = 6;
  std::size_t lattice_size = dimX * dimY;

  const int num_interactions = 50;
  const int walker_per_interactions = 100;
  const int num_lattices = num_interactions * walker_per_interactions;

  const int E_min = -2 * dimX * dimY;
  const int E_max = -E_min;

  const int len_single_histogram = E_max - E_min + 1;
  const int len_total_histogram = num_interactions * len_single_histogram;

  const int num_intervals = 10;
  const int total_intervals = num_intervals * num_interactions;

  const int num_wl_iterations = 10000;

  const int THREADS = 128;
  const int BLOCKS = (num_lattices * dimY * dimX * 2 + THREADS - 1) / THREADS;
  const int BLOCKS_LATTICE = (num_lattices + THREADS - 1) / THREADS;

  std::random_device rd;
  const int seed = static_cast<int>(rd());

  std::default_random_engine generator(seed);

  std::uniform_real_distribution<float> distribution(0.0, 1.0);
  float prob = distribution(generator);

  wl::interval_results result{E_min, E_max, num_intervals, 1, 1.0F};

  // Lattice initialization
  wl::device_ptr<std::int8_t> d_lattice(dimX * dimY * num_lattices, 0);
  wl::device_ptr<float> d_init_probs_per_lattice(num_lattices, 0);

  wl::init_lattice<<<BLOCKS, THREADS>>>(
    d_lattice.data(), d_init_probs_per_lattice.data(), dimX, dimY, num_lattices, seed, walker_per_interactions);

  // Interaction initialization
  wl::device_ptr<std::int8_t> d_interactions(num_interactions * dimX * dimY * 2, 0);
  wl::init_interactions<<<BLOCKS, THREADS>>>(d_interactions.data(), lattice_size, num_interactions, seed + 1, prob);

  // Lattice Offsets
  wl::device_ptr<int> d_offset_lattice(num_lattices, 0);
  wl::device_ptr<int> d_offset_found_lattice(total_intervals, 0);
  wl::init_offsets_lattice<<<BLOCKS_LATTICE, THREADS>>>(d_offset_lattice.data(), dimX, dimY, num_lattices);
  wl::init_offsets_lattice<<<BLOCKS_LATTICE, THREADS>>>(d_offset_found_lattice.data(), dimX, dimY, total_intervals);

  // Places to store intervals
  wl::device_ptr<std::int8_t> d_found_interval_lattice(dimX * dimY * num_intervals * num_interactions, 0);
  wl::device_ptr<int> d_flag_found_interval(num_intervals * num_interactions, 0);
  wl::device_ptr<int> d_energies_found_interval(num_intervals * num_interactions, 0);

  // Histogram initialization
  wl::device_ptr<unsigned long long> d_histogram(len_total_histogram, 0);

  // Seed offset initialization
  wl::device_ptr<unsigned long> d_seed_offset(num_lattices, 0);

  // Energy initialization
  wl::device_ptr<int> d_energy(num_lattices, 0);
  wl::device_ptr<int> d_test_energy(num_lattices, 0);

  wl::calc_energy_periodic_boundary<<<BLOCKS_LATTICE, THREADS>>>(d_energy.data(),
                                                                 d_lattice.data(),
                                                                 d_interactions.data(),
                                                                 d_offset_lattice.data(),
                                                                 dimX,
                                                                 dimY,
                                                                 num_lattices,
                                                                 walker_per_interactions);

  for (int i = 0; i < 200; i++) {
    wl::wang_landau_pre_run<<<BLOCKS_LATTICE, THREADS>>>(d_energy.data(),
                                                         d_flag_found_interval.data(),
                                                         d_lattice.data(),
                                                         d_found_interval_lattice.data(),
                                                         d_histogram.data(),
                                                         d_seed_offset.data(),
                                                         d_interactions.data(),
                                                         d_offset_lattice.data(),
                                                         E_min,
                                                         E_max,
                                                         num_wl_iterations,
                                                         dimX,
                                                         dimY,
                                                         seed,
                                                         num_intervals,
                                                         result.len_interval,
                                                         num_lattices,
                                                         walker_per_interactions,
                                                         wl::boundary::periodic);

    wl::calc_energy_periodic_boundary<<<BLOCKS_LATTICE, THREADS>>>(d_test_energy.data(),
                                                                   d_lattice.data(),
                                                                   d_interactions.data(),
                                                                   d_offset_lattice.data(),
                                                                   dimX,
                                                                   dimY,
                                                                   num_lattices,
                                                                   walker_per_interactions);

    // Copy energies and compare
    std::vector<int> h_energy(num_lattices, 0);
    std::vector<int> h_test_energy(num_lattices, 0);

    d_energy.device_to_host(h_energy);
    d_test_energy.device_to_host(h_test_energy);

    BOOST_CHECK_EQUAL_COLLECTIONS(h_energy.begin(), h_energy.end(), h_test_energy.begin(), h_test_energy.end());
  }

  wl::calc_energy_periodic_boundary<<<BLOCKS_LATTICE, THREADS>>>(d_energies_found_interval.data(),
                                                                 d_found_interval_lattice.data(),
                                                                 d_interactions.data(),
                                                                 d_offset_found_lattice.data(),
                                                                 dimX,
                                                                 dimY,
                                                                 total_intervals,
                                                                 num_intervals);

  std::vector<int> h_energies_found_interval(total_intervals, 0);
  d_energies_found_interval.device_to_host(h_energies_found_interval);

  std::vector<std::int8_t> h_found_lattice(total_intervals * dimX * dimY);
  d_found_interval_lattice.device_to_host(h_found_lattice);

  for (int i = 0; i < num_interactions; i++) {
    for (int j = 0; j < num_intervals; j++) {
      int offset = (i * num_intervals + j) * dimX * dimY;

      if (h_found_lattice[offset] == 0) {
        continue;
      }

      BOOST_CHECK(h_energies_found_interval[i * num_intervals + j] >= result.h_start[j] &&
                  h_energies_found_interval[i * num_intervals + j] <= result.h_end[j]);
    }
  }
}

BOOST_AUTO_TEST_CASE(multiple_iterations_open)
{
  const int dimX = 6;
  const int dimY = 6;
  std::size_t lattice_size = dimX * dimY;

  const int num_interactions = 50;
  const int walker_per_interactions = 100;
  const int num_lattices = num_interactions * walker_per_interactions;

  const int E_min = -2 * dimX * dimY;
  const int E_max = -E_min;

  const int len_single_histogram = E_max - E_min + 1;
  const int len_total_histogram = num_interactions * len_single_histogram;

  const int num_intervals = 10;
  const int total_intervals = num_intervals * num_interactions;

  const int num_wl_iterations = 10000;

  const int THREADS = 128;
  const int BLOCKS = (num_lattices * dimY * dimX * 2 + THREADS - 1) / THREADS;
  const int BLOCKS_LATTICE = (num_lattices + THREADS - 1) / THREADS;

  std::random_device rd;
  const int seed = static_cast<int>(rd());

  std::default_random_engine generator(seed);

  std::uniform_real_distribution<float> distribution(0.0, 1.0);
  float prob = distribution(generator);

  wl::interval_results result{E_min, E_max, num_intervals, 1, 1.0F};

  // Lattice initialization
  wl::device_ptr<std::int8_t> d_lattice(dimX * dimY * num_lattices, 0);
  wl::device_ptr<float> d_init_probs_per_lattice(num_lattices, 0);

  wl::init_lattice<<<BLOCKS, THREADS>>>(
    d_lattice.data(), d_init_probs_per_lattice.data(), dimX, dimY, num_lattices, seed, walker_per_interactions);

  // Interaction initialization
  wl::device_ptr<std::int8_t> d_interactions(num_interactions * dimX * dimY * 2, 0);
  wl::init_interactions<<<BLOCKS, THREADS>>>(d_interactions.data(), lattice_size, num_interactions, seed + 1, prob);

  // Lattice Offsets
  wl::device_ptr<int> d_offset_lattice(num_lattices, 0);
  wl::device_ptr<int> d_offset_found_lattice(total_intervals, 0);
  wl::init_offsets_lattice<<<BLOCKS_LATTICE, THREADS>>>(d_offset_lattice.data(), dimX, dimY, num_lattices);
  wl::init_offsets_lattice<<<BLOCKS_LATTICE, THREADS>>>(d_offset_found_lattice.data(), dimX, dimY, total_intervals);

  // Places to store intervals
  wl::device_ptr<std::int8_t> d_found_interval_lattice(dimX * dimY * num_intervals * num_interactions, 0);
  wl::device_ptr<int> d_flag_found_interval(num_intervals * num_interactions, 0);
  wl::device_ptr<int> d_energies_found_interval(num_intervals * num_interactions, 0);

  // Histogram initialization
  wl::device_ptr<unsigned long long> d_histogram(len_total_histogram, 0);

  // Seed offset initialization
  wl::device_ptr<unsigned long> d_seed_offset(num_lattices, 0);

  // Energy initialization
  wl::device_ptr<int> d_energy(num_lattices, 0);
  wl::device_ptr<int> d_test_energy(num_lattices, 0);

  wl::calc_energy_open_boundary<<<BLOCKS_LATTICE, THREADS>>>(d_energy.data(),
                                                                 d_lattice.data(),
                                                                 d_interactions.data(),
                                                                 d_offset_lattice.data(),
                                                                 dimX,
                                                                 dimY,
                                                                 num_lattices,
                                                                 walker_per_interactions);

  for (int i = 0; i < 200; i++) {
    wl::wang_landau_pre_run<<<BLOCKS_LATTICE, THREADS>>>(d_energy.data(),
                                                         d_flag_found_interval.data(),
                                                         d_lattice.data(),
                                                         d_found_interval_lattice.data(),
                                                         d_histogram.data(),
                                                         d_seed_offset.data(),
                                                         d_interactions.data(),
                                                         d_offset_lattice.data(),
                                                         E_min,
                                                         E_max,
                                                         num_wl_iterations,
                                                         dimX,
                                                         dimY,
                                                         seed,
                                                         num_intervals,
                                                         result.len_interval,
                                                         num_lattices,
                                                         walker_per_interactions,
                                                         wl::boundary::open);

    wl::calc_energy_open_boundary<<<BLOCKS_LATTICE, THREADS>>>(d_test_energy.data(),
                                                                   d_lattice.data(),
                                                                   d_interactions.data(),
                                                                   d_offset_lattice.data(),
                                                                   dimX,
                                                                   dimY,
                                                                   num_lattices,
                                                                   walker_per_interactions);

    // Copy energies and compare
    std::vector<int> h_energy(num_lattices, 0);
    std::vector<int> h_test_energy(num_lattices, 0);

    d_energy.device_to_host(h_energy);
    d_test_energy.device_to_host(h_test_energy);

    BOOST_CHECK_EQUAL_COLLECTIONS(h_energy.begin(), h_energy.end(), h_test_energy.begin(), h_test_energy.end());
  }

  wl::calc_energy_open_boundary<<<BLOCKS_LATTICE, THREADS>>>(d_energies_found_interval.data(),
                                                                 d_found_interval_lattice.data(),
                                                                 d_interactions.data(),
                                                                 d_offset_found_lattice.data(),
                                                                 dimX,
                                                                 dimY,
                                                                 total_intervals,
                                                                 num_intervals);

  std::vector<int> h_energies_found_interval(total_intervals, 0);
  d_energies_found_interval.device_to_host(h_energies_found_interval);

  std::vector<std::int8_t> h_found_lattice(total_intervals * dimX * dimY);
  d_found_interval_lattice.device_to_host(h_found_lattice);

  for (int i = 0; i < num_interactions; i++) {
    for (int j = 0; j < num_intervals; j++) {
      int offset = (i * num_intervals + j) * dimX * dimY;

      if (h_found_lattice[offset] == 0) {
        continue;
      }

      BOOST_CHECK(h_energies_found_interval[i * num_intervals + j] >= result.h_start[j] &&
                  h_energies_found_interval[i * num_intervals + j] <= result.h_end[j]);
    }
  }
}

/**
 * In this test we check whether we obtain the correct histogram after a long enough prerun for the lattice
 * size 4x4.
 *
 */
BOOST_AUTO_TEST_CASE(energy_range_periodic)
{
  const int dimX = 4;
  const int dimY = 4;
  const std::size_t lattice_size = dimX * dimY;

  const int num_interactions = 1;
  const int walker_per_interactions = 100;
  const int num_lattices = num_interactions * walker_per_interactions;

  const int E_min = -2 * dimX * dimY;
  const int E_max = -E_min;

  const int len_single_histogram = E_max - E_min + 1;
  const int len_total_histogram = num_interactions * len_single_histogram;

  const int num_intervals = 5;
  const int total_intervals = num_intervals * num_interactions;

  const int num_wl_iterations = 200000;

  const int THREADS = 128;
  const int BLOCKS = (num_lattices * dimY * dimX * 2 + THREADS - 1) / THREADS;
  const int BLOCKS_LATTICE = (num_lattices + THREADS - 1) / THREADS;

  std::random_device rd;
  const int seed = static_cast<int>(rd());

  float prob = 0;

  wl::interval_results result{E_min, E_max, num_intervals, 1, 1.0F};

  // Lattice initialization
  wl::device_ptr<std::int8_t> d_lattice(dimX * dimY * num_lattices, 0);
  wl::device_ptr<float> d_init_probs_per_lattice(num_lattices, 0);

  wl::init_lattice<<<BLOCKS, THREADS>>>(
    d_lattice.data(), d_init_probs_per_lattice.data(), dimX, dimY, num_lattices, seed, walker_per_interactions);

  // Interaction initialization
  wl::device_ptr<std::int8_t> d_interactions(num_interactions * dimX * dimY * 2, 0);
  wl::init_interactions<<<BLOCKS, THREADS>>>(d_interactions.data(), lattice_size, num_interactions, seed + 1, prob);

  // Lattice Offsets
  wl::device_ptr<int> d_offset_lattice(num_lattices, 0);
  wl::device_ptr<int> d_offset_found_lattice(total_intervals, 0);
  wl::init_offsets_lattice<<<BLOCKS_LATTICE, THREADS>>>(d_offset_lattice.data(), dimX, dimY, num_lattices);
  wl::init_offsets_lattice<<<BLOCKS_LATTICE, THREADS>>>(d_offset_found_lattice.data(), dimX, dimY, total_intervals);

  // Places to store intervals
  wl::device_ptr<std::int8_t> d_found_interval_lattice(dimX * dimY * num_intervals * num_interactions, 0);
  wl::device_ptr<int> d_flag_found_interval(num_intervals * num_interactions, 0);

  // Histogram initialization
  wl::device_ptr<unsigned long long> d_histogram(len_total_histogram, 0);

  // Seed offset initialization
  wl::device_ptr<unsigned long> d_seed_offset(num_lattices, 0);

  // Energy initialization
  wl::device_ptr<int> d_energy(num_lattices, 0);

  wl::calc_energy_periodic_boundary<<<BLOCKS_LATTICE, THREADS>>>(d_energy.data(),
                                                                 d_lattice.data(),
                                                                 d_interactions.data(),
                                                                 d_offset_lattice.data(),
                                                                 dimX,
                                                                 dimY,
                                                                 num_lattices,
                                                                 walker_per_interactions);

  wl::wang_landau_pre_run<<<BLOCKS_LATTICE, THREADS>>>(d_energy.data(),
                                                       d_flag_found_interval.data(),
                                                       d_lattice.data(),
                                                       d_found_interval_lattice.data(),
                                                       d_histogram.data(),
                                                       d_seed_offset.data(),
                                                       d_interactions.data(),
                                                       d_offset_lattice.data(),
                                                       E_min,
                                                       E_max,
                                                       num_wl_iterations,
                                                       dimX,
                                                       dimY,
                                                       seed,
                                                       num_intervals,
                                                       result.len_interval,
                                                       num_lattices,
                                                       walker_per_interactions,
                                                       wl::boundary::periodic);

  std::vector<unsigned long long> h_histogram(len_total_histogram, 0);

  d_histogram.device_to_host(h_histogram);

  for (int i = 0; i < len_total_histogram; i++) {
    int energy = E_min + i;
    if (energy % 4 == 0) {
      if (energy == -28 || energy == 28) {
        BOOST_CHECK(h_histogram[i] == 0);
      } else {
        BOOST_CHECK(h_histogram[i] != 0);
      }
    } else {
      BOOST_CHECK(h_histogram[i] == 0);
    }
  }
}

BOOST_AUTO_TEST_CASE(energy_range_open)
{
  const int dimX = 4;
  const int dimY = 4;
  const std::size_t lattice_size = dimX * dimY;

  const int num_interactions = 1;
  const int walker_per_interactions = 100;
  const int num_lattices = num_interactions * walker_per_interactions;

  const int E_min = -2 * dimX * dimY;
  const int E_max = -E_min;

  const int len_single_histogram = E_max - E_min + 1;
  const int len_total_histogram = num_interactions * len_single_histogram;

  const int num_intervals = 5;
  const int total_intervals = num_intervals * num_interactions;

  const int num_wl_iterations = 200000;

  const int THREADS = 128;
  const int BLOCKS = (num_lattices * dimY * dimX * 2 + THREADS - 1) / THREADS;
  const int BLOCKS_LATTICE = (num_lattices + THREADS - 1) / THREADS;

  std::random_device rd;
  const int seed = static_cast<int>(rd());

  float prob = 0;

  wl::interval_results result{E_min, E_max, num_intervals, 1, 1.0F};

  // Lattice initialization
  wl::device_ptr<std::int8_t> d_lattice(dimX * dimY * num_lattices, 0);
  wl::device_ptr<float> d_init_probs_per_lattice(num_lattices, 0);

  wl::init_lattice<<<BLOCKS, THREADS>>>(
    d_lattice.data(), d_init_probs_per_lattice.data(), dimX, dimY, num_lattices, seed, walker_per_interactions);

  // Interaction initialization
  wl::device_ptr<std::int8_t> d_interactions(num_interactions * dimX * dimY * 2, 0);
  wl::init_interactions<<<BLOCKS, THREADS>>>(d_interactions.data(), lattice_size, num_interactions, seed + 1, prob);

  // Lattice Offsets
  wl::device_ptr<int> d_offset_lattice(num_lattices, 0);
  wl::device_ptr<int> d_offset_found_lattice(total_intervals, 0);
  wl::init_offsets_lattice<<<BLOCKS_LATTICE, THREADS>>>(d_offset_lattice.data(), dimX, dimY, num_lattices);
  wl::init_offsets_lattice<<<BLOCKS_LATTICE, THREADS>>>(d_offset_found_lattice.data(), dimX, dimY, total_intervals);

  // Places to store intervals
  wl::device_ptr<std::int8_t> d_found_interval_lattice(dimX * dimY * num_intervals * num_interactions, 0);
  wl::device_ptr<int> d_flag_found_interval(num_intervals * num_interactions, 0);

  // Histogram initialization
  wl::device_ptr<unsigned long long> d_histogram(len_total_histogram, 0);

  // Seed offset initialization
  wl::device_ptr<unsigned long> d_seed_offset(num_lattices, 0);

  // Energy initialization
  wl::device_ptr<int> d_energy(num_lattices, 0);

  wl::calc_energy_open_boundary<<<BLOCKS_LATTICE, THREADS>>>(d_energy.data(),
                                                                 d_lattice.data(),
                                                                 d_interactions.data(),
                                                                 d_offset_lattice.data(),
                                                                 dimX,
                                                                 dimY,
                                                                 num_lattices,
                                                                 walker_per_interactions);

  wl::wang_landau_pre_run<<<BLOCKS_LATTICE, THREADS>>>(d_energy.data(),
                                                       d_flag_found_interval.data(),
                                                       d_lattice.data(),
                                                       d_found_interval_lattice.data(),
                                                       d_histogram.data(),
                                                       d_seed_offset.data(),
                                                       d_interactions.data(),
                                                       d_offset_lattice.data(),
                                                       E_min,
                                                       E_max,
                                                       num_wl_iterations,
                                                       dimX,
                                                       dimY,
                                                       seed,
                                                       num_intervals,
                                                       result.len_interval,
                                                       num_lattices,
                                                       walker_per_interactions,
                                                       wl::boundary::open);

  std::vector<unsigned long long> h_histogram(len_total_histogram, 0);

  d_histogram.device_to_host(h_histogram);

  for (int i = 0; i < len_total_histogram; i++) {
    int energy = E_min + i;
    if (energy % 2 == 0 && energy >= (-2*dimX*dimY + dimX + dimY) && energy <= (2*dimX*dimY - dimX - dimY) && energy != 2*dimX*dimY - dimX - dimY - 2 && energy != -2*dimX*dimY + dimX + dimY + 2) {
      BOOST_CHECK(h_histogram[i] != 0);
    } else {
      BOOST_CHECK(h_histogram[i] == 0);
    }
  }
}

BOOST_AUTO_TEST_SUITE_END()
