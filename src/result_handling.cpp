#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <tuple>

#include "result_handling.hpp"
#include "utils.hpp"

namespace {
/**
 * @brief Retrieves the log(g) data from a set of walkers per interval over all intervals of a disoder sample.
 *
 * This function processes the input `h_logG` data and organizes it into intervals,
 * based on `h_start` and `h_end` arrays. It only stores non-zero values into a map
 * per interval, which is then used for further rescaling and stitching.
 *
 * @param h_logG A vector of doubles representing log(g) values from walkers.
 * @param h_start A vector of integers representing the start energies of intervals.
 * @param h_end A vector of integers representing the end energies of intervals.
 * @param options The run options for the number of intervals and walkers per interval.
 *
 * @return A vector of maps, where each map contains unique energy values and corresponding log(g) values per interval.
 */
auto get_logG_data(const std::vector<double>& h_logG,
                   const std::vector<int>& h_start,
                   const std::vector<int>& h_end,
                   const wl::mainrun_options& options)
{
  int index_h_log_g = 0;
  const auto num_intervals = static_cast<std::size_t>(options.num_intervals);

  // Store the results of the first walker for each interval as they are averaged already
  std::vector<std::map<int, double>> interval_data(num_intervals);

  for (std::size_t i = 0; i < num_intervals; i++) {
    int len_int = h_end[i] - h_start[i] + 1;

    for (int j = 0; j < options.walker_per_interval; j++) {
      if (j == 0) {
        for (int k = 0; k < len_int; k++) {
          int key = h_start[i] + k;
          double value = h_logG[static_cast<std::size_t>(index_h_log_g)];

          if (value != 0) {
            interval_data[i][key] =
              value; // Store the non-zero value with its key at correct map object according to interval
          }

          index_h_log_g += 1;
        }
      } else {
        index_h_log_g += len_int;
      }
    }
  }

  return interval_data;
}

/**
 * @brief Rescales the log(g) data by subtracting the minimum value in each interval.
 *
 * This function iterates through each interval's data and finds the minimum
 * log(g) value. It then subtracts this minimum from all values in the interval
 * to ensure the smallest value in each interval becomes zero.
 *
 * @param interval_data A vector of maps, where each map contains the log(g) values for an interval over energy keys.
 * @param options The run options.
 *
 * @return A vector of maps with rescaled values for each interval.
 */
auto rescale_by_minimum(std::vector<std::map<int, double>>& interval_data, const wl::mainrun_options& options)
{
  const auto num_intervals = static_cast<std::size_t>(options.num_intervals);
  //
  // finding minimum per interval
  std::vector<double> min_values(num_intervals, std::numeric_limits<double>::max());
  for (std::size_t i = 0; i < num_intervals; i++) {
    for (const auto& key_value_pair : interval_data[i]) {
      if (key_value_pair.second < min_values[i]) {
        min_values[i] = key_value_pair.second;
      }
    }

    // If no non-zero value was found, reset to 0 (or any other default)
    if (min_values[i] == std::numeric_limits<double>::max()) {
      min_values[i] = 0;
    }
  }

  // rescaling by minimum
  for (std::size_t i = 0; i < num_intervals; i++) {
    for (auto& key_value_pair : interval_data[i]) {
      key_value_pair.second -= min_values[i]; // each interval has a zero value now
    }
  }

  return interval_data;
}

/**
 * @brief Finds the optimal key for stitching two input intervals based on their derivative differences.
 *
 * This function compares the energy values in two neighboring intervals and identifies under shared energy keys the
 * most appropriate stitching key that minimizes the derivative (linear approx of d(log(g))/dE) difference between
 * the intervals. This is used for stitching adjacent intervals together.
 *
 * @param current_interval A map containing the log(g) data for the current interval.
 * @param next_interval A map containing the log(g) data for the next interval.
 *
 * @return A tuple containing the key for stitching and the corresponding derivative difference.
 */
auto find_stitching_keys(const std::map<int, double>& current_interval, const std::map<int, double>& next_interval)
{
  int min_key = std::numeric_limits<int>::max();
  double min_diff = std::numeric_limits<double>::max();

  auto it1 = current_interval.begin();
  auto it2 = next_interval.begin();

  while (it1 != current_interval.end() && it2 != next_interval.end()) {
    if (it1->first == it2->first) {
      double diff1 = (std::next(it1)->second - it1->second) / (std::next(it1)->first - it1->first);
      double diff2 = (std::next(it2)->second - it2->second) / (std::next(it2)->first - it2->first);
      double diff_between_intervals = std::abs(diff1 - diff2);

      if (diff_between_intervals < min_diff) {
        min_diff = diff_between_intervals;
        min_key = it1->first;
      }
      ++it1;
      ++it2;
    } else if (it1->first < it2->first) {
      ++it1;
    } else {
      ++it2;
    }
  }

  return std::make_tuple(min_key, min_diff);
}

/**
 * @brief Calculates stitching points between intervals.
 *
 * This function calculates the optimal stitching points between all adjacent intervals
 * by comparing their derivatives and identifying points where the data can be
 * smoothly connected.
 *
 * @param interval_data A vector of maps, where each map contains log(g) values for an interval over energy keys.
 * @param options The run options object.
 *
 * @return A vector of tuples containing the index of the interval and the key for stitching.
 */
auto calculate_stitching_points(const std::vector<std::map<int, double>>& interval_data,
                                const wl::mainrun_options& options)
{
  const auto num_intervals = static_cast<std::size_t>(options.num_intervals);
  std::vector<std::tuple<int, int>> stitching_keys;

  for (std::size_t i = 0; i < num_intervals - 1; i++) {
    int absolute_min_key = std::numeric_limits<int>::max();
    double min_derivative = std::numeric_limits<double>::max();
    int interval_index = std::numeric_limits<int>::max();

    const auto& current_interval = interval_data[i];

    for (std::size_t j = i + 1; j < num_intervals; j++) {
      const auto& next_interval = interval_data[j];

      auto [min_key, deriv] = find_stitching_keys(current_interval, next_interval);

      if (min_key < std::numeric_limits<int>::max()) {
        if (deriv < min_derivative) {
          absolute_min_key = min_key;
          min_derivative = deriv;
          interval_index = static_cast<int>(j);
        }
      } else {
        break;
      }
    }
    if (interval_index < std::numeric_limits<int>::max()) {
      stitching_keys.emplace_back(interval_index, absolute_min_key);
    }
  }

  return stitching_keys;
}

/**
 * @brief Rescales the intervals for concatenation by applying shifts to ensure continuity.
 *
 * This function applies shifts to the log(g) data in each interval based on the stitching
 * keys. The shift is applied to align the values of adjacent intervals.
 *
 * @param interval_data A vector of maps containing log(g) data for each interval.
 * @param stitching_keys A vector of tuples containing the optimal stitching points.
 */
void rescale_intervals_for_concatenation(std::vector<std::map<int, double>>& interval_data,
                                         const std::vector<std::tuple<int, int>>& stitching_keys)
{
  for (size_t i = 0; i < stitching_keys.size(); ++i) {
    int e_concat = std::get<1>(stitching_keys[i]);
    auto next_interval = static_cast<std::size_t>(std::get<0>(stitching_keys[i]));

    auto idx_in_preceding_interval = interval_data[0].find(e_concat);

    if (i != 0) {
      idx_in_preceding_interval =
        interval_data[static_cast<std::size_t>(std::get<0>(stitching_keys[i - 1]))].find(e_concat);
    }

    auto idx_in_following_interval = interval_data[next_interval].find(e_concat);

    if (idx_in_preceding_interval == interval_data[i].end() ||
        idx_in_following_interval == interval_data[next_interval].end()) {
      throw std::runtime_error("stitching energy " + std::to_string(e_concat) +
                               " not found in one of the intervals which may be caused by non overlapping "
                               "intervals which can not be normalized properly.");
    }

    double shift_val =
      idx_in_preceding_interval->second -
      idx_in_following_interval->second; // difference by which the following interval results get affinely shifted

    // Apply the shift to all values in the following interval
    for (auto& [key, value] : interval_data[next_interval]) {
      value += shift_val;
    }
  }
}

/**
 * @brief Filters the log(g) data based on a threshold of the energy key.
 *
 * This function filters a map of energy values and log(g) data by selecting entries
 * whose keys are either less than or greater than a specified threshold.
 *
 * @param data A map of energy keys to log(g) values.
 * @param threshold The threshold key for filtering the energy range.
 * @param less_than A boolean flag indicating whether to filter keys less than the threshold.
 *
 * @return A filtered map containing only the selected entries.
 */
auto filter_map_by_key(const std::map<int, double>& data, int threshold, bool less_than)
{
  std::map<int, double> filtered_map;
  for (const auto& [key, value] : data) {
    if ((less_than && key < threshold) || (!less_than && key >= threshold)) {
      filtered_map[key] = value;
    }
  }
  return filtered_map;
}

/**
 * @brief Cuts overlapping parts of the histogram data at stitching keys.
 *
 * This function uses the calculated stitching keys to filter overlapping parts of
 * the histogram data, ensuring only unique energy values are retained for each interval.
 *
 * @param interval_data A vector of maps containing log(g) data for each interval.
 * @param stitching_keys A vector of tuples containing the stitching points for each interval.
 *
 * @return A vector of maps with the filtered data.
 */
auto cut_overlapping_histogram_parts(const std::vector<std::map<int, double>>& interval_data,
                                     const std::vector<std::tuple<int, int>>& stitching_keys)
{
  std::vector<std::map<int, double>> filtered_data;

  // Filter the first interval
  if (!stitching_keys.empty()) {
    int first_energy = std::get<1>(stitching_keys[0]);
    auto filtered_map = filter_map_by_key(interval_data[0], first_energy, true);
    if (!filtered_map.empty()) {
      filtered_data.push_back(filtered_map);
    }
  }

  // Filter the intermediate intervals
  for (std::size_t i = 1; i < stitching_keys.size(); ++i) {
    int energy = std::get<1>(stitching_keys[i]);
    int energy_prev = std::get<1>(stitching_keys[i - 1]);
    auto previous_interval_index = static_cast<std::size_t>(std::get<0>(stitching_keys[i - 1]));
    auto filtered_map = filter_map_by_key(interval_data[previous_interval_index], energy, true);
    filtered_map = filter_map_by_key(filtered_map, energy_prev, false);

    if (!filtered_map.empty()) {
      filtered_data.push_back(filtered_map);
    }
  }

  // Filter the last interval
  int last_energy = std::get<1>(stitching_keys.back());
  auto last_interval_index = static_cast<std::size_t>(std::get<0>(stitching_keys.back()));
  auto final_filtered_map = filter_map_by_key(interval_data[last_interval_index], last_energy, false);
  if (!final_filtered_map.empty()) {
    filtered_data.push_back(final_filtered_map);
  }

  return filtered_data;
}

/**
 * @brief Calculates the log of the sum of exponentials of the log(g) values across all intervals.
 *
 * This function computes the log-sum-exp for a set of data. It computes
 * the sum of the exponentials of the log(g) values after rescaling them based on the
 * maximum value in the dataset and adds the maximum at the end again which circumvents overflow issues.
 *
 * @param data A vector of maps containing log(g) data.
 *
 * @return The log of the sum of exponentials of the data values.
 */
double log_sum_exp(const std::vector<std::map<int, double>>& data)
{
  double maxVal = -std::numeric_limits<double>::infinity();

  // Get the maximum value to rescale for numerical reason
  for (const auto& data_map : data) {
    for (const auto& data_pair : data_map) {
      if (data_pair.second > maxVal) {
        maxVal = data_pair.second;
      }
    }
  }

  // Calculate sum of exp(values - maxVal)
  double sumExp = 0.0;
  for (const auto& data_map : data) {
    for (const auto& data_pair : data_map) {
      sumExp += std::exp(data_pair.second - maxVal);
    }
  }

  // rescale by maxVal to retrieve original log sum exp without overflow issues
  return maxVal + std::log(sumExp);
}

/**
 * @brief Rescales the log(g) values by gauging it's corresponding log-sum-exp value to the configuration space volume.
 *
 * This function rescales the log(g) values by applying a transformation
 * based on the log-sum-exp and the dimensions `dimX` and `dimY`. This ensures proper scaling
 * of the values for high-temperature calculations.
 *
 * @param data A vector of maps containing log(g) data.
 * @param dimX The X dimension for rescaling.
 * @param dimY The Y dimension for rescaling.
 */
void rescale_map_values(std::vector<std::map<int, double>>& data, double dimX, double dimY)
{
  double offset = log_sum_exp(data);
  double log2XY = std::log(2) * dimX * dimY;

  for (auto& data_map : data) {
    for (auto& data_pair : data_map) {
      data_pair.second = data_pair.second + log2XY - offset;
    }
  }
}
} // namespace close

void wl::result_handling_stitched_histogram(const mainrun_options& options,
                                            const std::vector<double>& h_logG,
                                            const std::vector<int>& h_start,
                                            const std::vector<int>& h_end,
                                            const std::string_view result_path,
                                            const std::string_view timestamp_group_name,
                                            const std::string_view git_info,
                                            int int_id)
{
  auto interval_data = get_logG_data(h_logG, h_start, h_end, options);
  auto rescaled_data = rescale_by_minimum(interval_data, options);
  auto stitching_keys = calculate_stitching_points(rescaled_data, options);

  std::vector<std::tuple<int, int>> real_stitching_keys;

  for (std::size_t i = 0; i < stitching_keys.size(); i++) {
    int energy_key = std::get<1>(stitching_keys[i]);

    bool check_intersection = true;

    for (std::size_t j = i + 1; j < stitching_keys.size(); j++) {
      if (std::get<1>(stitching_keys[j]) <= energy_key) {
        check_intersection = false;
      }
    }
    if (check_intersection) {
      real_stitching_keys.push_back(stitching_keys[i]);
    }
  }

  // Vector to store the smallest values for each key
  std::vector<std::tuple<int, int>> smallest_values;

  // Initialize the first key-value tuple
  int current_key = std::get<0>(real_stitching_keys[0]);
  int current_min_value = std::get<1>(real_stitching_keys[0]);

  // Iterate over the key-value tuples (assuming sorted by key)
  for (const auto& [key, value] : real_stitching_keys) {
    // If the key changes, store the smallest value for the previous key
    if (key != current_key) {
      smallest_values.emplace_back(current_key, current_min_value); // Save the smallest value
      current_key = key;
      current_min_value = value; // reset for the new key
    } else {
      // If the key is the same, update the minimum value
      if (value < current_min_value) {
        current_min_value = value;
      }
    }
  }

  // Don't forget to add the last key-value pair after the loop ends
  smallest_values.emplace_back(current_key, current_min_value);

  rescale_intervals_for_concatenation(rescaled_data, smallest_values);

  std::vector<std::map<int, double>> cut_data = cut_overlapping_histogram_parts(rescaled_data, smallest_values);

  rescale_map_values(
    cut_data, options.X, options.Y); // rescaling for high temperature interpretation of partition function

  write_results(cut_data, options, int_id, result_path, timestamp_group_name, git_info);
}
