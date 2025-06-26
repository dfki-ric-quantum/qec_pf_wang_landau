#ifndef FILE_IO_HPP_
#define FILE_IO_HPP_

#include <cstdint>
#include <map>
#include <vector>
#include <string_view>

#include <H5Gpublic.h>
#include <H5Tpublic.h>

#include "program_options.hpp"

namespace wl {

/**
 * @brief Writes the histogram results from the prerun to an HDF5 file.
 *
 * This function processes the histogram data and writes it to an HDF5 file under a specified path.
 * It organizes the data into groups corresponding to each disorder sample.
 * The written histogram captures only binary information whether the given energy is reachable for the model, i.e. we
 * are not writing ull to storage but unsigned chars.
 * All prerun results are stored in the same HDF5 file for a given set of run parameters but logically seperated into
 * datasets: Histogam, Interaction and Lattice.
 *
 * @param h_histograms A vector of unsigned long long integers containing the histogram data for all disorder samples
 * @param options A prerun options object containing all prerun parameters
 * @param len_histogram The length of each histogram given by maximal energy range
 * @param file_id The file identifier for the already created/open HDF5 file
 *
 * @throws std::runtime_error If any HDF5 function fails
 *
 */
void write_prerun_histogram_results(const std::vector<unsigned long long>& h_histograms,
                                    const wl::prerun_options& options,
                                    std::size_t len_histogram,
                                    hid_t file_id);

/**
 * @brief Writes the interaction samples from the prerun to an HDF5 file.
 *
 * This function stores the interaction configurations in an HDF5 file
 * under the specified path.
 * All prerun results are stored in the same HDF5 file for a given set of run parameters but logically seperated into
 * datasets: Histogam, Interaction and Lattice.
 *
 * @param h_interactions A vector of signed characters containing the interaction configurations for all disorder
 * samples
 * @param options A prerun options object containing all prerun parameters
 * @param file_id The file identifier for the already created/open HDF5 file
 *
 * @throws std::runtime_error If any HDF5 function fails
 *
 */
void write_prerun_interaction_results(const std::vector<signed char>& h_interactions,
                                      const wl::prerun_options& options,
                                      hid_t file_id);

/**
 * @brief Writes the prerun lattice configurations and corresponding energies to an HDF5 file.
 *
 * This function organizes the lattice data and corresponding energies in datasets for each disorder sample as group
 * and writes them to an HDF5 file under the specified path.
 * All prerun results are stored in the same HDF5 file for a given set of run parameters but logically seperated into
 * datasets: Histogam, Interaction and Lattice.
 *
 * @param h_lattices A vector of signed characters representing the lattice configurations over all intervals and
 * disorder samples
 * @param h_energies A vector of integers representing the energies associated with the lattices
 * @param options A prerun options object containing all prerun parameters
 * @param file_id The file identifier for the already created/open HDF5 file
 *
 * @throws std::runtime_error If any HDF5 function fails
 *
 */
void write_prerun_lattice_results(const std::vector<signed char>& h_lattices,
                                  const std::vector<int>& h_energies,
                                  const wl::prerun_options& options,
                                  hid_t file_id);

/**
 * @brief Writes all prerun results (histograms, interactions, lattices) to an HDF5 file.
 *
 * This function stores histogram, interaction and lattice results for all disorder samples
 * to an HDF5 file under the specified path. The different results are seperated into different datasets: Histogam,
 * Interaction and Lattice.
 *
 * @param h_histograms A vector of unsigned long long integers containing the histogram data
 * @param h_interactions A vector of signed characters containing the interaction data
 * @param h_lattices A vector of signed characters containing the lattice data
 * @param h_energies A vector of integers representing the energies
 * @param options A prerun options object containing all prerun parameters
 * @param len_histogram The length of each histogram given by the maximal energy range
 * @param file_path The path to the directory where the HDF5 file will be created
 *
 * @throws std::runtime_error If any HDF5 function fails
 *
 */
void write_prerun_results(const std::vector<unsigned long long>& h_histograms,
                          const std::vector<signed char>& h_interactions,
                          const std::vector<signed char>& h_lattices,
                          const std::vector<int>& h_energies,
                          const wl::prerun_options& options,
                          std::size_t len_histogram,
                          const std::string_view file_path);

/**
 * @brief Reads histogram data from a prerun HDF5 file into a destination map.
 *
 * This function retrieves histogram data from an HDF5 file and stores it in a map
 * where keys represent disorder sample offset and values contain the histogram data.
 *
 * @param h_histograms_dest A map to store histogram data for each disorder sample
 * @param file_path The path to the HDF5 file containing histogram data
 * @param len_histogram The length of each histogram given by maximal energy range
 *
 * @throws std::runtime_error If any HDF5 function fails
 *
 */
void read_prerun_histograms(std::map<int, std::vector<unsigned char>>& h_histograms_dest,
                            const std::string_view file_path,
                            std::size_t len_histogram);

/**
 * @brief Reads interaction data from a prerun HDF5 file into a destination map.
 *
 * This function retrieves interaction data from an HDF5 file and stores it in a map
 * where keys represent disorder sample IDs and values contain the interaction data.
 *
 * @param h_interactions_dest A map to store interaction data for each disorder sample
 * @param file_path The path to the HDF5 file containing interaction data
 * @param options A prerun options object containing all prerun parameters
 *
 * @throws std::runtime_error If any HDF5 function fails
 *
 */
void read_prerun_interactions(std::map<int, std::vector<signed char>>& h_interactions_dest,
                              const std::string_view file_path,
                              const wl::prerun_options options);

void read_prerun_interactions(std::map<int, std::vector<signed char>>& h_interactions_dest,
                              const std::string_view file_path,
                              const wl::mainrun_options& options);

/**
 * @brief Reads lattice configurations and associated energy data from a prerun HDF5 file.
 *
 * This function retrieves lattice configurations and energy data from an HDF5 file
 * and stores them in a nested map, where the outer keys represent disorder sample offsets
 * and the inner keys represent energy eigenvalues.
 *
 * @param h_lattices_dest A nested map to store lattice configurations and energies for the disorder sample offset
 * @param options A prerun options object containing all prerun parameters
 * @param file_path The path to the HDF5 file containing lattice and energy data
 *
 * @throws std::runtime_error If any HDF5 function fails
 *
 */
void read_prerun_lattices(std::map<int, std::map<int, std::vector<signed char>>>& h_lattices_dest,
                          const wl::prerun_options options,
                          const std::string_view file_path);

void read_prerun_lattices(std::map<int, std::map<int, std::vector<signed char>>>& h_lattices_dest,
                          const wl::mainrun_options& options,
                          const std::string_view file_path);

/**
 * @brief Computes energy spectrum information from prerun histograms.
 *
 * This function processes the provided histograms for each interaction and computes the minimum
 * and maximum energy values, populates the energy spectrum, and calculates offsets and lengths
 * for each histogram segment.
 *
 * @param[out] h_expected_energy_spectrum Vector to store the combined energy spectrum with cutted boundaries.
 * @param[out] h_offset_energy_spectrum Vector to store the starting offset of each energy spectrum segment.
 * @param[out] h_len_energy_spectrum Vector to store the length of each energy spectrum segment.
 * @param[out] E_min Vector to store the minimum energy value for each interaction.
 * @param[out] E_max Vector to store the maximum energy value for each interaction.
 * @param[in, out] total_len_energy_spectrum The running total length of the energy spectrum (updated in-place).
 * @param[in] prerun_histograms Map containing histograms for each interaction.
 * @param[in] options The mainrun options object containing run parameters.
 *
 * @throws std::runtime_error if a histogram does not contain a single 1.
 */
void get_energy_spectrum_information_from_prerun_results(
  std::vector<signed char>& h_expected_energy_spectrum,
  std::vector<int>& h_offset_energy_spectrum,
  std::vector<int>& h_len_energy_spectrum,
  std::vector<int>& E_min,
  std::vector<int>& E_max,
  int total_len_energy_spectrum,
  const std::map<int, std::vector<unsigned char>>& prerun_histograms,
  const wl::mainrun_options& options);

/**
 * @brief Retrieves interactions from prerun results and appends them to the provided interaction vector.
 *
 * This function processes a map of prerun interactions and retrieves the interaction data for a
 * specified number of interactions, as defined in the mainrun options. If an interaction key is missing
 * from the map, the function throws an exception.
 *
 * @param[out] h_interactions A vector to store the retrieved interaction data. The function appends
 *                            the interactions to this vector.
 * @param[in] prerun_interactions A map where keys represent disorder samples and values are
 *                                vectors containing interaction terms.
 * @param[in] options A mainrun option object representing the run parameter.
 *
 * @throws std::runtime_error if an disorder sample offset key is not found in `prerun_interactions` or if read
 * interaction dim is not compatible with lattice dim.
 *
 * @note The function assumes that `options.num_interactions` specifies the exact number of interactions
 *       that should be retrieved from the map. Missing keys will result in an exception.
 */
void get_interaction_from_prerun_results(std::vector<signed char>& h_interactions,
                                         const std::map<int, std::vector<signed char>>& prerun_interactions,
                                         const wl::mainrun_options& options);

/**
 * @brief Extracts lattice configurations from prerun results based on energy intervals and appends them to a vector.
 *
 * This function retrieves lattice configurations from a nested map structure (`prerun_lattices`) for each
 * disorder sample (`disorder_key`) and appends them to `h_lattices` if their energy falls within the specified
 * intervals.
 *
 * @param[out] h_lattices A vector to which the selected lattice configurations are appended.
 * @param[in] prerun_lattices A nested map storing lattice configurations, where:
 *                            - The outer key (`int`) is the disorder sample key.
 *                            - The inner map's key (`int`) represents the energy values of the spin configs.
 *                            - The inner map's value (`std::vector<signed char>`) stores the lattice configuration.
 * @param[in] h_start A vector specifying the start of each energy interval for all disorder keys over all intervals.
 * @param[in] h_end A vector specifying the end of each energy interval for all disorder keys over all intervals.
 * @param[in] options A mainrun options object structure containign the run parameter.
 *
 * @throws std::runtime_error if a lattice configuration with a suitable energy is not found for any interval or if no
 * lattices are found for a disorder sample.
 *
 */
void get_lattice_from_prerun_results(std::vector<std::int8_t>& h_lattices,
                                     const std::map<int, std::map<int, std::vector<signed char>>>& prerun_lattices,
                                     const std::vector<int>& h_start_int,
                                     const std::vector<int>& h_end_int,
                                     const wl::mainrun_options& options);

} // wl

#endif // FILE_IO_HPP_
