#ifndef UTILS_HPP_
#define UTILS_HPP_

#include <chrono>
#include <map>
#include <string>
#include <string_view>

#include "git_info/git_info.hpp"
#include "interval.hpp"
#include "program_options.hpp"

namespace wl {
/**
 * Log levels to distinguish in output
 */
enum struct log_level
{
  info,
  warn,
  error
};

/**
 * Simple log function.
 *
 * @param msg The log message to print to clog
 * @param level The log level
 */
void log(std::string_view msg, log_level level = log_level::info);

/**
 * Helper function to log intervals
 *
 * @param The interval results
 */
void log_intervals(const interval_results& ivr);

/**
 * Get seed from the OS's (pseudo-)random device
 *
 * @return A (pseudo-) random seed
 */
[[nodiscard]] int get_seed_from_os();

/**
 * Build the output path for the pre-run
 *
 * @param options The program options passed to the prerun
 * @return The path as string
 */
[[nodiscard]] auto build_prerun_path(const prerun_options& options) -> std::string;

/**
 * Build path to read mainrun data from
 *
 * @param options The options passed to the mainrun
 * @return The full file path as string
 */
[[nodiscard]] auto build_mainrun_path(const mainrun_options& options) -> std::string;

/**
 * Create directory (including subdirectories) for the prerun input from mainrun parameters
 *
 * @param path The full path of all subdirectories to create
 */
void create_directories(std::string_view path);

/**
 * Build the output path for the mainrun results
 *
 * @param options The options passed to the mainrun
 * @return The full file path as string
 */
[[nodiscard]] auto build_mainrun_result_path(const mainrun_options& options) -> std::string;

/**
 * @brief Writes Wang Landau results into an HDF5 file.
 *
 * This function flattens the input result data (a vector of maps per non overlapping interval) and writes it into a
 * dataset in an HDF5 file. It organizes the results into groups based on the disorder ID, run seed, and timestamp.
 * Additionally, the function stores the Git version information as an attribute in the dataset to track the code
 * version used for the simulation.
 *
 * @param result_data A vector of maps where each map contains key-value pairs of integers (energy keys)
 *                    and doubles (log(g) values).
 * @param options The run options containing the run seed.
 * @param disorder_id The disorder group identifier, used to organize results in the HDF5 file.
 * @param file_path The directory path where the HDF5 result file will be created or opened.
 * @param timestamp_group_name The name of the timestamp group used to organize the dataset within the file.
 * @param git_info The Git code version to be stored as an attribute in the dataset.
 *
 * @throws std::runtime_error If any errors occur during the file creation, dataset writing, or attribute writing
 * process.
 */
void write_results(const std::vector<std::map<int, double>>& result_data,
                   const mainrun_options& options,
                   int disorder_id,
                   const std::string_view file_path,
                   const std::string_view timestamp_group_name,
                   const std::string_view git_info);

/**
 * Returns git version ifo string
 *
 * @return The last git tag, distance from it, commit hash and if the repo has changes
 */

[[nodiscard]] inline std::string get_git_version()
{
  return std::string{wl::git::version};
}

/**
 * @brief Generates a timestamp string suitable for use as a group name in hdf5 file.
 *
 * This function creates a timestamp string based on the current system time.
 * The resulting string is formatted to replace spaces with underscores ('_')
 * and has no newline characters at the end. The output is suitable for use as
 * HDF5 group names.
 *
 * @return A `std::string` representing the sanitized timestamp group name.
 */
[[nodiscard]] auto get_timestamp_group_name() -> std::string;

/**
 * @brief Safely casts a long long to an int after verifying its range.
 *
 * This function checks whether the given long long value lies within the range
 * of an int. If the value is within the valid range, it performs the cast.
 * Otherwise, it throws a std::out_of_range exception.
 *
 * @param value The long long value to be cast to int.
 * @return int The safely casted value as an int.
 *
 * @throws std::out_of_range If the value is outside the valid range for int.
 */
int cast_safely_longlong_to_int(long long value);

} // wl

#endif // UTILS_HPP_
