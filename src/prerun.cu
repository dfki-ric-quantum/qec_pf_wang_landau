#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

#include "cuda_utils.cuh"
#include "file_io.hpp"
#include "interval.hpp"
#include "program_options.hpp"
#include "utils.hpp"
#include "wl.cuh"

const unsigned int THREADS = 128; // Number of CUDA threads per block

int main(int argc, char** argv)
{
  const auto [options, cont] = wl::parse_prerun_options(argc, argv);

  if (!cont) {
    return EXIT_SUCCESS;
  }

  const int e_min = -2 * options.X * options.Y; // Minimum energy, assuming coupling is bounded by 1
  const int e_max = -e_min;
  const wl::interval_results interval_result{e_min, e_max, options.num_intervals, 1, 1.0F};
  wl::log_intervals(interval_result);

  const auto total_walker = static_cast<unsigned int>(options.num_interactions * options.walker_per_interactions);
  const auto total_intervals = static_cast<unsigned int>(options.num_interactions) * options.num_intervals;

  const auto len_histogram = static_cast<unsigned int>(e_max - e_min + 1);
  const auto total_histogram = static_cast<unsigned int>(options.num_interactions) * len_histogram;
  const auto lattice_size = static_cast<unsigned int>(options.X * options.Y);

  try {
    wl::device_ptr<unsigned long long> d_histogram(total_histogram, 0);
    wl::device_ptr<unsigned long> d_seed_offset(total_walker, 0);
    wl::device_ptr<std::int8_t> d_lattice(total_walker * lattice_size, 0);
    wl::device_ptr<float> d_probs(total_walker, 0);
    wl::device_ptr<std::int8_t> d_interactions(static_cast<unsigned int>(options.num_interactions) * lattice_size * 2, 0);
    wl::device_ptr<int> d_energy(total_walker, 0);
    wl::device_ptr<std::int8_t> d_found_interval_lattice(total_intervals * lattice_size, 0);
    wl::device_ptr<int> d_flag_found_interval(total_intervals, 0);
    wl::device_ptr<int> d_interval_energies(total_intervals, 0);
    wl::device_ptr<int> d_offset_lattice_per_walker(total_walker, 0);
    wl::device_ptr<int> d_offset_lattice_per_interval(total_intervals, 0);

    const auto BLOCKS_INIT = static_cast<unsigned int>((total_walker * lattice_size * 2 + THREADS - 1) / THREADS);
    const auto BLOCKS_ENERGY = (total_walker + THREADS - 1) / THREADS;
    const auto BLOCKS_INTERVAL = static_cast<unsigned int>(total_intervals + THREADS - 1) / THREADS;

    wl::init_interactions<<<BLOCKS_INIT, THREADS>>>(
      d_interactions.data(), lattice_size, options.num_interactions, options.seed, options.prob_interactions);

    // Application of non trivial error cycles
    switch (options.logical_error_type) {
      case 'h':
        wl::apply_x_horizontal_error<<<BLOCKS_INIT, THREADS>>>(
          d_interactions.data(), options.X, options.Y, options.num_interactions);
        wl::log("Applied horizontal error cycle.");
        break;
      case 'v':
        wl::apply_x_vertical_error<<<BLOCKS_INIT, THREADS>>>(
          d_interactions.data(), options.X, options.Y, options.num_interactions);
        wl::log("Applied vertical error cycle.");
        break;
      case 'c': // combines the vertical and horizonatl error cycle
        wl::apply_x_horizontal_error<<<BLOCKS_INIT, THREADS>>>(
          d_interactions.data(), options.X, options.Y, options.num_interactions);
        wl::apply_x_vertical_error<<<BLOCKS_INIT, THREADS>>>(
          d_interactions.data(), options.X, options.Y, options.num_interactions);
        wl::log("Applied both horizontal and vertical error cycle.");
        break;
      default: wl::log("Trivial error class."); break;
    }

    wl::init_lattice<<<BLOCKS_INIT, THREADS>>>(d_lattice.data(),
                                               d_probs.data(),
                                               options.X,
                                               options.Y,
                                               total_walker,
                                               options.seed,
                                               options.walker_per_interactions);

    wl::init_offsets_lattice<<<BLOCKS_ENERGY, THREADS>>>(
      d_offset_lattice_per_walker.data(), options.X, options.Y, total_walker);

    wl::init_offsets_lattice<<<BLOCKS_INTERVAL, THREADS>>>(
      d_offset_lattice_per_interval.data(), options.X, options.Y, total_intervals);

    wl::calc_energy(BLOCKS_ENERGY,
                    THREADS,
                    options.boundary,
                    d_energy.data(),
                    d_lattice.data(),
                    d_interactions.data(),
                    d_offset_lattice_per_walker.data(),
                    options.X,
                    options.Y,
                    total_walker,
                    options.walker_per_interactions);

    wl::wang_landau_pre_run<<<BLOCKS_ENERGY, THREADS>>>(d_energy.data(),
                                                        d_flag_found_interval.data(),
                                                        d_lattice.data(),
                                                        d_found_interval_lattice.data(),
                                                        d_histogram.data(),
                                                        d_seed_offset.data(),
                                                        d_interactions.data(),
                                                        d_offset_lattice_per_walker.data(),
                                                        e_min,
                                                        e_max,
                                                        options.num_iterations,
                                                        options.X,
                                                        options.Y,
                                                        options.seed,
                                                        options.num_intervals,
                                                        interval_result.len_interval,
                                                        total_walker,
                                                        options.walker_per_interactions,
                                                        options.boundary);

    wl::calc_energy(BLOCKS_INTERVAL,
                    THREADS,
                    options.boundary,
                    d_interval_energies.data(),
                    d_found_interval_lattice.data(),
                    d_interactions.data(),
                    d_offset_lattice_per_interval.data(),
                    options.X,
                    options.Y,
                    total_intervals,
                    static_cast<int>(options.num_intervals));

    if (!options.skip_output) {
      std::vector<int> h_interval_energies(total_intervals);
      std::vector<std::int8_t> h_interactions(
        static_cast<std::size_t>(options.X * options.Y * 2 * options.num_interactions));
      std::vector<std::int8_t> h_found_interval_lattice(static_cast<std::size_t>(options.X * options.Y) *
                                                        total_intervals);
      std::vector<unsigned long long> h_histogram(total_histogram);

      d_interval_energies.device_to_host(h_interval_energies);
      d_interactions.device_to_host(h_interactions);
      d_found_interval_lattice.device_to_host(h_found_interval_lattice);
      d_histogram.device_to_host(h_histogram);

      const std::string path = wl::build_prerun_path(options);

      wl::create_directories(path);

      wl::write_prerun_results(
        h_histogram, h_interactions, h_found_interval_lattice, h_interval_energies, options, len_histogram, path);
    }

    wl::log(std::format("Finished prerun for Lattice {}x{}, probability {}, error type {} and {} interactions",
                        options.X,
                        options.Y,
                        options.prob_interactions,
                        options.logical_error_type,
                        options.num_interactions));
  } catch (const std::exception& err) {
    wl::log(err.what(), wl::log_level::error);
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
