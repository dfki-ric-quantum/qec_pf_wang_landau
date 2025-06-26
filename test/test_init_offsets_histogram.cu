#include <boost/test/tools/old/interface.hpp>
#define BOOST_TEST_MODULE wl_tests

#include <random>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <thrust/reduce.h>
#include <vector>

#include <boost/test/unit_test.hpp>
#include <cuda_runtime.h>

#include "../src/cuda_utils.cuh"
#include "../src/interval.hpp"
#include "../src/wl.cuh"

BOOST_AUTO_TEST_SUITE(check_init_histogram_offsets)

/**
 * Tests whether histogram offsets coincide with the wanted values
 */
BOOST_AUTO_TEST_CASE(check_plausability)
{
  const int dimX = 10;
  const int dimY = 10;

  const int num_interactions = 10;
  const int num_intervals = 10;
  const int num_walker_per_interval = 5;
  const float overlap = 0.25;

  const int total_intervals = num_interactions * num_intervals;
  const int total_walker = total_intervals * num_walker_per_interval;

  int emin = -2 * dimX * dimY;
  int emax = -emin;

  std::vector<int> E_min(num_interactions, 0);
  std::vector<int> E_max(num_interactions, 0);

  std::random_device random_device;  // Seed generator
  std::mt19937 gen(random_device()); // Mersenne Twister engine

  const int e_interval_offset = 10;
  std::uniform_int_distribution<> maxs(emax - e_interval_offset, emax);
  std::uniform_int_distribution<> mins(emin, emin + e_interval_offset);

  // Get min and max energies per interaction
  for (unsigned int i = 0; i < num_interactions; i++) {
    E_min[i] = mins(gen);
    E_max[i] = maxs(gen);
  }

  // Generate intervals for all different energy spectrums
  std::vector<int> h_end_int;
  std::vector<int> h_start_int;
  std::vector<int> len_histogram_int;
  std::vector<int> len_interval_int;

  for (unsigned int i = 0; i < num_interactions; i++) {
    wl::interval_results result{E_min[i], E_max[i], num_intervals, num_walker_per_interval, overlap};

    h_end_int.insert(h_end_int.end(), result.h_end.begin(), result.h_end.end());

    h_start_int.insert(h_start_int.end(), result.h_start.begin(), result.h_start.end());

    len_histogram_int.push_back(static_cast<int>(result.len_histogram_over_all_walkers));
    len_interval_int.push_back(result.len_interval);
  }

  wl::device_ptr<int> d_len_histograms(len_histogram_int);
  wl::device_ptr<int> d_start_interval_energies(h_start_int);
  wl::device_ptr<int> d_end_interval_energies(h_end_int);

  wl::device_ptr<int> d_offset_histogram(total_walker, 0);

  wl::init_offsets_histogram<<<total_intervals, num_walker_per_interval>>>(d_offset_histogram.data(),
                                                                           d_start_interval_energies.data(),
                                                                           d_end_interval_energies.data(),
                                                                           d_len_histograms.data(),
                                                                           num_intervals,
                                                                           total_walker);

  std::vector<int> h_offset_histogram(total_walker);
  d_offset_histogram.device_to_host(h_offset_histogram);

  std::vector<int> real_offsets(total_walker, 0);

  int interaction_offset = 0;

  for (int i = 0; i < num_interactions; i++) {
    auto first_idx = static_cast<std::size_t>(i*num_intervals);
    int len_first_interval = h_end_int[first_idx] - h_start_int[first_idx] + 1;
    for (int j = 0; j < num_intervals; j++) {
      int interval_offset = j * num_walker_per_interval * len_first_interval;
      for (int k = 0; k < num_walker_per_interval; k++) {
        if (j != num_intervals - 1) {
          int final_offset = interaction_offset + interval_offset + k * len_first_interval;
          real_offsets[static_cast<std::size_t>(i * num_intervals * num_walker_per_interval + j * num_walker_per_interval + k)] = final_offset;
        } else {
          int len_last_interval = h_end_int[static_cast<std::size_t>((i + 1) * num_intervals - 1)]
                                  - h_start_int[static_cast<std::size_t>((i + 1) * num_intervals - 1)] + 1;
          int final_offset = interaction_offset + interval_offset + k * len_last_interval;
          real_offsets[static_cast<std::size_t>(i * num_intervals * num_walker_per_interval + j * num_walker_per_interval + k)] = final_offset;
        }
      }
    }
    interaction_offset += len_histogram_int[static_cast<std::size_t>(i)];
  }

  for (unsigned int i = 0; i < total_walker; i++) {
    BOOST_CHECK(h_offset_histogram[i] == real_offsets[i]);
  }
}

BOOST_AUTO_TEST_SUITE_END()
