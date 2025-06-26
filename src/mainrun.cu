#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <format>
#include <numbers>

#include <cub/cub.cuh>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>

#include "cuda_utils.cuh"
#include "file_io.hpp"
#include "program_options.hpp"
#include "result_handling.hpp"
#include "utils.hpp"
#include "wl.cuh"
#include "file_io.hpp"

namespace {
bool check_histogram_bounds(int offset, int length, int totalLength)
{
  if (offset + length > totalLength) {
    wl::log("Error: Copy range exceeds histogram bounds", wl::log_level::error);
    return false;
  }
  return true;
}
} //

int main(int argc, char** argv)
{
  int max_threads_per_block = 128;

  auto start = std::chrono::steady_clock::now();

  const auto [options, cont] = wl::parse_mainrun_options(argc, argv);

  if (!cont) {
    return EXIT_SUCCESS;
  }

  const int total_walker = options.num_interactions * options.num_intervals * options.walker_per_interval;
  const int total_intervals = options.num_interactions * options.num_intervals;
  const int walker_per_interactions = options.num_intervals * options.walker_per_interval;

  std::vector<std::int8_t> h_expected_energy_spectrum;
  std::vector<int> h_len_energy_spectrum;
  std::vector<int> h_offset_energy_spectrum;
  std::vector<int> E_min;
  std::vector<int> E_max;
  std::vector<int> h_end_int;
  std::vector<int> h_start_int;
  std::vector<int> len_histogram_int;
  std::vector<int> len_interval_int;
  std::vector<long long> h_offset_shared_log_G;
  std::vector<double> h_factor(total_walker, std::numbers::e);
  std::vector<int> h_cond_interactions(options.num_interactions);
  std::vector<int> h_offset_intervals(options.num_interactions + 1);

  std::map<int, std::vector<unsigned char>> prerun_histograms;

  int max_len_histogram = 4 * options.X * options.Y + 1;

  int total_len_energy_spectrum = 0;
  long long total_len_histogram = 0;

  std::string init_path = wl::build_mainrun_path(options);
  wl::read_prerun_histograms(prerun_histograms, init_path, static_cast<std::size_t>(max_len_histogram));

  wl::get_energy_spectrum_information_from_prerun_results(h_expected_energy_spectrum,
                                                          h_offset_energy_spectrum,
                                                          h_len_energy_spectrum,
                                                          E_min,
                                                          E_max,
                                                          total_len_energy_spectrum,
                                                          prerun_histograms,
                                                          options);

  for (int i = 0; i < options.num_interactions; i++) {
    wl::interval_results run_result{E_min[i],
                                    E_max[i],
                                    static_cast<unsigned int>(options.num_intervals),
                                    options.walker_per_interval,
                                    options.overlap_decimal};

    h_end_int.insert(h_end_int.end(), run_result.h_end.begin(), run_result.h_end.end());

    h_start_int.insert(h_start_int.end(), run_result.h_start.begin(), run_result.h_start.end());

    len_histogram_int.push_back(wl::cast_safely_longlong_to_int(run_result.len_histogram_over_all_walkers));

    len_interval_int.push_back(run_result.len_interval);
    total_len_histogram += run_result.len_histogram_over_all_walkers;

    wl::log(std::format("Last interval: {} - {}, other interval len: {}, num intervals: {}",
                        h_start_int.back(),
                        h_end_int.back(),
                        len_interval_int[i],
                        options.num_intervals));
  }

  int size_shared_log_G = 0;
  for (int i = 0; i < options.num_interactions; i++) {
    for (int j = 0; j < options.num_intervals; j++) {
      h_offset_shared_log_G.push_back(size_shared_log_G);
      size_shared_log_G += (h_end_int[i * options.num_intervals + j] - h_start_int[i * options.num_intervals + j] + 1);
    }
  }

  for (int i = 0; i < options.num_interactions; i++) {
    h_offset_intervals[i] = i * options.num_intervals;
  }
  h_offset_intervals[options.num_interactions] = total_intervals;

  try {
    wl::device_ptr<int> d_offset_energy_spectrum(h_offset_energy_spectrum);
    wl::device_ptr<int> d_len_energy_spectrum(h_len_energy_spectrum);
    wl::device_ptr<int> d_len_histograms(len_histogram_int);

    // Start end energies of the intervals
    wl::device_ptr<int> d_start(h_start_int);
    wl::device_ptr<int> d_end(h_end_int);

    // Histogram and G array
    wl::device_ptr<unsigned long long> d_H(total_len_histogram, 0);
    wl::device_ptr<double> d_logG(total_len_histogram, 0);

    wl::device_ptr<double> d_shared_logG(size_shared_log_G, 0);
    wl::device_ptr<long long> d_offset_shared_logG(h_offset_shared_log_G);

    // Offset histograms, lattice, seed_iterator
    wl::device_ptr<int> d_offset_histogram(total_walker);
    wl::device_ptr<int> d_offset_lattice(total_walker);
    wl::device_ptr<unsigned long> d_offset_iter(total_walker, 0);

    // f Factors for each walker
    wl::device_ptr<double> d_factor(h_factor);

    // Indices used for replica exchange later
    wl::device_ptr<int> d_indices(total_walker);

    // lattice, interactions
    wl::device_ptr<std::int8_t> d_lattice(total_walker * options.X * options.Y);
    wl::device_ptr<std::int8_t> d_interactions(options.num_interactions * options.X * options.Y * 2);

    // Hamiltonian of lattices
    wl::device_ptr<int> d_energy(total_walker, 0);

    // Flag for correct energy ranges
    wl::device_ptr<std::int8_t> d_flag_check_energies(total_walker, 0);

    // Binary indicator of energies were found or not
    wl::device_ptr<std::int8_t> d_expected_energy_spectrum(h_expected_energy_spectrum);

    // To catch energies which are outside of expected spectrum
    wl::device_ptr<int> d_newEnergies(total_walker);
    wl::device_ptr<int> d_foundNewEnergyFlag(total_walker);

    wl::device_ptr<std::int8_t> d_cond(total_intervals, 0);
    wl::device_ptr<int> d_cond_interactions(options.num_interactions, 0);

    wl::device_ptr<int> d_offset_intervals(h_offset_intervals);

    // Temp storage for wl::check_interactions_finished
    wl::device_tmp d_tmp_storage{};

    /*
    ----------------------------------------------
    ------------ Actual WL Starts Now ------------
    ----------------------------------------------
    */

    // Initialization of lattices, interactions, offsets and indices
    wl::init_offsets_lattice<<<total_intervals, options.walker_per_interval>>>(
      d_offset_lattice.data(), options.X, options.Y, total_walker);

    wl::init_offsets_histogram<<<total_intervals, options.walker_per_interval>>>(d_offset_histogram.data(),
                                                                                 d_start.data(),
                                                                                 d_end.data(),
                                                                                 d_len_histograms.data(),
                                                                                 options.num_intervals,
                                                                                 total_walker);

    wl::init_indices<<<total_intervals, options.walker_per_interval>>>(d_indices.data(), total_walker);

    std::vector<std::int8_t> h_interactions;

    std::map<int, std::vector<std::int8_t>> prerun_interactions;
    wl::read_prerun_interactions(prerun_interactions, init_path, options);

    wl::get_interaction_from_prerun_results(h_interactions, prerun_interactions, options);

    d_interactions.host_to_device(h_interactions);

    std::map<int, std::map<int, std::vector<std::int8_t>>> prerun_lattices;
    wl::read_prerun_lattices(prerun_lattices, options, init_path);

    std::vector<std::int8_t> h_lattices;

    wl::get_lattice_from_prerun_results(h_lattices, prerun_lattices, h_start_int, h_end_int, options);

    d_lattice.host_to_device(h_lattices);

    wl::calc_energy(total_intervals,
                    options.walker_per_interval,
                    options.boundary,
                    d_energy.data(),
                    d_lattice.data(),
                    d_interactions.data(),
                    d_offset_lattice.data(),
                    options.X,
                    options.Y,
                    total_walker,
                    walker_per_interactions);

    wl::check_energy_ranges<<<total_intervals, options.walker_per_interval>>>(
      d_flag_check_energies.data(), d_energy.data(), d_start.data(), d_end.data(), total_walker);

    int max_energy_check = 0;

    thrust::device_ptr<int8_t> d_check_energy_pointer(d_flag_check_energies.data());
    thrust::device_ptr<int8_t> max_flag_check_energies_ptr =
      thrust::max_element(d_check_energy_pointer, d_check_energy_pointer + total_walker);
    max_energy_check = *max_flag_check_energies_ptr;

    if (max_energy_check == 1) {
      throw std::runtime_error("Calculated energies do not match intervals");
    }

    // Stop timing
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    wl::log(std::format("Execution time before Wang Landau has started: {} seconds.", elapsed.count()));

    double max_factor = std::numbers::e;

    int max_newEnergyFlag = 0;

    int block_count = wl::cast_safely_longlong_to_int((total_len_histogram + max_threads_per_block - 1) / max_threads_per_block);

    long long wang_landau_counter = 1;

    while (max_factor - exp(options.beta) > 1e-10) // hardcoded precision for abort condition
    {
      wl::wang_landau<<<total_intervals, options.walker_per_interval>>>(d_lattice.data(),
                                                                        d_interactions.data(),
                                                                        d_H.data(),
                                                                        d_logG.data(),
                                                                        d_factor.data(),
                                                                        d_energy.data(),
                                                                        d_newEnergies.data(),
                                                                        d_offset_iter.data(),
                                                                        d_foundNewEnergyFlag.data(),
                                                                        d_start.data(),
                                                                        d_end.data(),
                                                                        d_offset_histogram.data(),
                                                                        d_offset_lattice.data(),
                                                                        d_cond.data(),
                                                                        d_expected_energy_spectrum.data(),
                                                                        d_offset_energy_spectrum.data(),
                                                                        d_cond_interactions.data(),
                                                                        options.X,
                                                                        options.Y,
                                                                        options.num_iterations,
                                                                        total_walker,
                                                                        options.num_intervals,
                                                                        walker_per_interactions,
                                                                        options.seed,
                                                                        options.boundary);

      // get max of found new energy flag array to condition break and update the
      // histogram file with value in new energy array
      thrust::device_ptr<int> d_newEnergyFlag_ptr(d_foundNewEnergyFlag.data());
      thrust::device_ptr<int> max_newEnergyFlag_ptr =
        thrust::max_element(d_newEnergyFlag_ptr, d_newEnergyFlag_ptr + total_walker);
      max_newEnergyFlag = *max_newEnergyFlag_ptr;

      // If flag shows new energies get the device arrays containing these to the
      // host, update histogram file and print error message.
      if (max_newEnergyFlag != 0) {
        wl::log("Error: Found new energy", wl::log_level::error);
        return EXIT_FAILURE;
      }

      wl::check_histogram<<<total_intervals, options.walker_per_interval>>>(d_H.data(),
                                                                            d_factor.data(),
                                                                            d_cond.data(),
                                                                            d_offset_histogram.data(),
                                                                            d_end.data(),
                                                                            d_start.data(),
                                                                            d_expected_energy_spectrum.data(),
                                                                            d_offset_energy_spectrum.data(),
                                                                            d_cond_interactions.data(),
                                                                            options.alpha,
                                                                            total_walker,
                                                                            walker_per_interactions,
                                                                            options.num_intervals);

      wl::calc_average_log_g<<<block_count, max_threads_per_block>>>(d_shared_logG.data(),
                                                                     d_len_histograms.data(),
                                                                     d_logG.data(),
                                                                     d_start.data(),
                                                                     d_end.data(),
                                                                     d_expected_energy_spectrum.data(),
                                                                     d_cond.data(),
                                                                     d_offset_histogram.data(),
                                                                     d_offset_energy_spectrum.data(),
                                                                     d_offset_shared_logG.data(),
                                                                     d_cond_interactions.data(),
                                                                     options.num_interactions,
                                                                     options.walker_per_interval,
                                                                     options.num_intervals,
                                                                     wl::cast_safely_longlong_to_int(total_len_histogram));

      wl::redistribute_g_values<<<block_count, max_threads_per_block>>>(d_logG.data(),
                                                                        d_shared_logG.data(),
                                                                        d_len_histograms.data(),
                                                                        d_start.data(),
                                                                        d_end.data(),
                                                                        d_cond_interactions.data(),
                                                                        d_offset_histogram.data(),
                                                                        d_cond.data(),
                                                                        d_offset_shared_logG.data(),
                                                                        options.num_intervals,
                                                                        options.walker_per_interval,
                                                                        options.num_interactions,
                                                                        wl::cast_safely_longlong_to_int(total_len_histogram));

      d_shared_logG.fill(0);

      wl::reset_d_cond<<<options.num_interactions, options.num_intervals>>>(
        d_cond.data(), d_factor.data(), total_intervals, options.beta, options.walker_per_interval);

      wl::check_interactions_finished(d_cond_interactions.data(),
                                      d_tmp_storage,
                                      d_cond.data(),
                                      d_offset_intervals.data(),
                                      options.num_intervals,
                                      options.num_interactions);

      // get max factor over walkers for abort condition of while loop
      thrust::device_ptr<double> d_factor_ptr(d_factor.data());
      thrust::device_ptr<double> max_factor_ptr = thrust::max_element(d_factor_ptr, d_factor_ptr + total_walker);
      max_factor = *max_factor_ptr;

      if (wang_landau_counter % options.replica_exchange_offset == 0) {
        wl::replica_exchange<<<total_intervals, options.walker_per_interval>>>(d_offset_lattice.data(),
                                                                               d_energy.data(),
                                                                               d_indices.data(),
                                                                               d_offset_iter.data(),
                                                                               d_start.data(),
                                                                               d_end.data(),
                                                                               d_offset_histogram.data(),
                                                                               d_cond_interactions.data(),
                                                                               d_logG.data(),
                                                                               true,
                                                                               options.seed,
                                                                               options.num_intervals,
                                                                               walker_per_interactions);

        wl::replica_exchange<<<total_intervals, options.walker_per_interval>>>(d_offset_lattice.data(),
                                                                               d_energy.data(),
                                                                               d_indices.data(),
                                                                               d_offset_iter.data(),
                                                                               d_start.data(),
                                                                               d_end.data(),
                                                                               d_offset_histogram.data(),
                                                                               d_cond_interactions.data(),
                                                                               d_logG.data(),
                                                                               false,
                                                                               options.seed,
                                                                               options.num_intervals,
                                                                               walker_per_interactions);
      }

      auto now = std::chrono::steady_clock::now();
      auto elapsed_seconds = std::chrono::duration_cast<std::chrono::seconds>(now - start).count();

      if (elapsed_seconds >= options.time_limit) {
        wl::log("Time limit reached. Terminating execution.", wl::log_level::warn);
        break;
      }
    }

    d_cond_interactions.device_to_host(h_cond_interactions);

    std::vector<int> h_offset_histogram(d_offset_histogram.size());
    d_offset_histogram.device_to_host(h_offset_histogram);

    const auto result_path =  wl::build_mainrun_result_path(options);
    wl::create_directories(result_path);

    for (int i = 0; i < options.num_interactions; i++) {
      if (h_cond_interactions[static_cast<std::size_t>(i)] == -1) {
        int offset_of_interaction_histogram = h_offset_histogram[static_cast<std::size_t>(i * options.num_intervals * options.walker_per_interval)];

        int len_of_interaction_histogram = len_histogram_int[static_cast<std::size_t>(i)];

        if (!check_histogram_bounds(
              offset_of_interaction_histogram, len_of_interaction_histogram, wl::cast_safely_longlong_to_int(total_len_histogram))) {
          wl::log("Error: Out of bounds histogram.", wl::log_level::error);
          return EXIT_FAILURE;
        }

        std::vector<double> h_logG(len_of_interaction_histogram);
        d_logG.device_to_host(h_logG, offset_of_interaction_histogram, len_of_interaction_histogram);

        std::vector<int> run_start(
          h_start_int.begin() + i * options.num_intervals,
          h_start_int.begin() +
            (i + 1) * options.num_intervals); // stores start energies of intervals of currently handled interaction

        std::vector<int> run_end(
          h_end_int.begin() + i * options.num_intervals,
          h_end_int.begin() +
            (i + 1) * options.num_intervals); // stores end energies of intervals of currently handled interaction

        wl::result_handling_stitched_histogram(
          options, h_logG, run_start, run_end, result_path, wl::get_timestamp_group_name(), wl::get_git_version(), i); // reduced result dump with X, Y needed for rescaling
      }
    }

  } catch (const std::exception& err) {
    wl::log(err.what(), wl::log_level::error);
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
