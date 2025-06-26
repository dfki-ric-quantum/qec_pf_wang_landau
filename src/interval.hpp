#ifndef INTERVAL_HPP_
#define INTERVAL_HPP_

#include <vector>

namespace wl {

/**
 * @brief This struct stores key information of the interval splitting for the Wang Landau algorithm.
 *
 * This struct is used to keep track of interval start and end points, the histogram lengths over all participating Wang
 * Landau walkers, and indicidual interval length.
 */
struct interval_results
{
  std::vector<int> h_start;                 /**< Vector storing the minimal energy per interval */
  std::vector<int> h_end;                   /**< Vector storing the maximal energy per interval */
  long long len_histogram_over_all_walkers; /**< The histogram length over all Wang Landau walkers */
  int len_interval; /**< The length of all first n-1 intervals. The last interval per interaction may have a difference
                       length.  */

  /**
   * The constructor.
   * @brief constructs an interval_results object representing a non disjoint cover of an energy interval for specified
   * interval parameters.
   *
   * Initializes minimal and maximal energies per interval, interval length and histogram length based on the provided
   * energy bounds, the number of intervals, the number of walkers, and the overlap between intervals.
   * The interval length calculation is based on the condition: len_interval + overlap * len_interval * (num_intervals -
   * 1) = e_range
   *
   * @param e_min Minimum energy for current model
   * @param e_max Maximum energy for current model
   * @param num_intervals Num intervals to split the energy range into
   * @param num_walkers Number of walkers per interval
   * @param overlap The overlap of neighboring intervals in decimal representation
   */
  interval_results(int e_min, int e_max, unsigned int num_intervals, int num_walkers, float overlap);
};
} // namespace wl

#endif // INTERVAL_HPP_
