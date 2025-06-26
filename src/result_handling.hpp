#ifndef RESULT_HANDLING_HPP_
#define RESULT_HANDLING_HPP_

#include "program_options.hpp"
#include <string>
#include <vector>

namespace wl {
/**
 * @brief Dumps the Wang Landau results for a specific disorder sample.
 *
 * This function processes the input log(g) data, rescaling it, calculating optimal
 * stitching points, and finally writing the results to the specified output location.
 *
 * It involves several steps including retrieving the log(g) data, rescaling it by the
 * minimum value, calculating stitching points, rescaling intervals for concatenation,
 * and cutting overlapping histogram parts.
 *
 * @param options The run options for the processing.
 * @param h_logG A vector of doubles representing log(g) values.
 * @param h_start A vector of integers representing the starting indices of intervals.
 * @param h_end A vector of integers representing the ending indices of intervals.
 * @param result_path The path where the results should be written.
 * @param timestamp_group_name The timestamp group name for output hierarchy in hdf5 file.
 * @param git_info The Git information for output attribute in hdf5 file.
 * @param int_id the disorder sample group name for output hierarchy in hdf5 file.
 */
void result_handling_stitched_histogram(const mainrun_options& options,
                                        const std::vector<double>& h_logG,
                                        const std::vector<int>& h_start,
                                        const std::vector<int>& h_end,
                                        const std::string_view result_path,
                                        const std::string_view timestamp_group_name,
                                        const std::string_view git_info,
                                        int int_id);
} //  wl

#endif // RESULT_HANDLING_HPP_
