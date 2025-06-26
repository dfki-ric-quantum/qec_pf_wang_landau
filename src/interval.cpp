#include <stdexcept>

#include "interval.hpp"

wl::interval_results::interval_results(int e_min, int e_max, unsigned int num_intervals, int num_walkers, float overlap)
  : h_start(num_intervals)
  , h_end(num_intervals)
  , len_histogram_over_all_walkers(0)
{
  if (overlap > 1 || overlap < 0) {
    throw std::invalid_argument("Overlap is out of bounds.");
  }

  const int e_range = e_max - e_min + 1;

  // len interval truncation towards zero may be relaxable but it is a safe option to find an appropriate interval cover
  len_interval = static_cast<int>(static_cast<float>(e_range) / (1.0F + overlap * static_cast<float>(num_intervals - 1)));

  // step size must be truncated toward zero to avoid to run out of energy range with specififed interval count
  const int step_size = static_cast<int>(overlap * static_cast<float>(len_interval));

  int start_interval = e_min;

  for (std::size_t i = 0; i < num_intervals; ++i) {
    h_start[i] = start_interval;

    if (i < num_intervals - 1) {
      h_end[i] = start_interval + len_interval - 1;
      len_histogram_over_all_walkers += num_walkers * len_interval;
    } else {
      h_end[i] = e_max;
      len_histogram_over_all_walkers += num_walkers * (e_max - h_start[i] + 1);
    }

    if (h_start[i] >= h_end[i]) {
      throw std::invalid_argument("The interval arguments create empty intervals.");
    }

    start_interval += step_size;
  }
}
