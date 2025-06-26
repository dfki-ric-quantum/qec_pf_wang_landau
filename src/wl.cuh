#ifndef WL_CUH_
#define WL_CUH_

#include <cstdint>
#include <cuda_runtime.h>

#include "boundary.hpp"
#include "cuda_utils.cuh"

namespace wl {
/**
 * @brief Calculates the energy with periodic boundary conditions for all spin configurations based
 * on the correspond interactions.
 *
 * @param d_energy Storage for energy values
 * @param d_lattice All spin configurations
 * @param d_interactions All interaction configurations
 * @param d_offset_lattice Offset per single spin configuration
 * @param dim_X X dimension of single spin configuration
 * @param dim_Y Y dimension of single spin configuration
 * @param num_lattices Total number of lattices in d_lattice
 * @param walker_per_interactions Number of walker per interaction
 */
__global__ void calc_energy_periodic_boundary(int* d_energy,
                                              const signed char* d_lattice,
                                              const signed char* d_interactions,
                                              const int* d_offset_lattice_per_walker,
                                              int dim_X,
                                              int dim_Y,
                                              std::size_t num_lattices,
                                              int walker_per_interactions);

/**
 * @brief Calculates the energy wth open boundary conditions for all spin configurations
 * based on the corresponding interactions.
 *
 * @param d_energy Storage for energy values
 * @param d_lattice All spin configurations
 * @param d_interactions All interaction configurations
 * @param d_offset_lattice Offset per single spin configuration
 * @param dim_X X dimension of single spin configuration
 * @param dim_Y Y dimension of single spin configuration
 * @param num_lattices Total number of lattices in d_lattice
 * @param walker_per_interactions Number of walker per interaction
 */
__global__ void calc_energy_open_boundary(int* d_energy,
                                          const signed char* d_lattice,
                                          const signed char* d_interactions,
                                          const int* d_offset_lattice,
                                          int dimX,
                                          int dimY,
                                          std::size_t num_lattices,
                                          int walker_per_interactions);

/**
 * @brief Calculates the energy for all spin configurations based on the correspond interactions.
 *
 * @param blocks Number of blocks per grid to launch the kernel with
 * @param threads Number of threads per block to launch the kernel with
 * @param boundary Boundary conditions. 0 -> periodic, 1 -> open
 * @param d_energy Storage for energy values
 * @param d_lattice All spin configurations
 * @param d_interactions All interaction configurations
 * @param d_offset_lattice Offset per single spin configuration
 * @param dim_X X dimension of single spin configuration
 * @param dim_Y Y dimension of single spin configuration
 * @param num_lattices Total number of lattices in d_lattice
 * @param walker_per_interactions Number of walker per interaction
 */
void calc_energy(unsigned int blocks,
                 unsigned int threads,
                 wl::boundary boundary,
                 int* d_energy,
                 const signed char* d_lattice,
                 const signed char* d_interactions,
                 const int* d_offset_lattice,
                 int dimX,
                 int dimY,
                 std::size_t num_lattices,
                 int walker_per_interactions);

/**
 * @brief Inititalizes the interactions for num_interactions many disorder samples for a given error probability.
 *
 * The interactions between spins are getting initialized here with a given probability to flip the interaction strength
 * from plus to minus one. Num interactions many disorder samples get initialized together and written to the
 * interactions array. The random sampling uses the seed parameter handed as parameter to the kernel.
 *
 * @param interactions The array storing the spin interaction terms
 * @param lattice_size Total size of the Ising spin lattice
 * @param num_interactions The amount of disorder samples to initialize
 * @param seed Random seed
 * @param prob Probability to flip an interaction term
 */
__global__ void init_interactions(int8_t* interactions,
                                  size_t lattice_size,
                                  int num_interactions,
                                  int seed,
                                  float prob);

/**
 * @brief This kernel flips all vertical interactions along the horizontal direction on the lattice for all disorder
 * samples.
 *
 * The up interactions of the first row of Ising spins for each disorder sample get flipped.
 * This error cycle represents a non trivial error class on the torus.
 *
 * @param interactions Stores the spin interactions with applied error cycle
 * @param lattice_dim_x The dimension in vertical direction for the Ising spin lattice
 * @param lattice_dim_y The dimension in horizontal direction for the Ising spin lattice
 * @param num_interactions The amount of disorder samples to initialize
 */
__global__ void apply_x_horizontal_error(signed char* interactions,
                                         int lattice_dim_x,
                                         int lattice_dim_y,
                                         int num_interactions);

/**
 * @brief This kernel flips all horizontal interactions along the vertical direction on the lattice for all disorder
 * samples.
 *
 * The left interactions of the first column of Ising spins for each disorder sample gets flipped.
 * This error cycle represents a non trivial error class on the torus.
 *
 * @param interactions Stores the spin interactions with applied error cycle
 * @param lattice_dim_x The dimension in vertical direction for the Ising spin lattice
 * @param lattice_dim_y The dimension in horizontak direction for the Ising spin lattice
 * @param num_interactions The amount of disorder samples to initialize
 */
__global__ void apply_x_vertical_error(signed char* interactions,
                                       int lattice_dim_x,
                                       int lattice_dim_y,
                                       int num_interactions);

/**
 * Initialize all lattice spins as -1 or 1.
 *
 * First sample probabilities for each lattice, then use this probability
 * to sample spins as -1 or 1 depending on the prob. This yields a more
 * diverse range of lattice configurations with larger energy range.
 *
 * @param d_lattice Device array for lattice spins
 * @param d_init_probs_per_lattice Device array to store probs for each lattice
 * @param dimX X dimension of single spin lattice
 * @param dimY Y dimension of single spin lattice
 * @param num_lattices Number of lattices contained in d_lattice
 * @param seed Random number seed
 * @param walker_per_interactions Number of walker for each interaction
 */
__global__ void init_lattice(signed char* d_lattice,
                             float* d_init_probs_per_lattice,
                             int dimX,
                             int dimY,
                             std::size_t num_lattices,
                             int seed,
                             int walker_per_interactions);

/**
 * @brief Initializes an array with indices for the beginning of each Ising lattice.
 *
 * As each Ising lattice contains dim_x times dim_y many spins, to access the n-th lattice one has to find the dim_x
 * times dim_y many spins starting at index n-1 time dim_x times dim_y when n is bigger zero. These indices to start
 * accessing the spin lattices are getting initialized here for each lattice.
 *
 * @param offset_lattice Array storing the initial index of each Ising spin lattice
 * @param lattice_dim_x The x/vertical dimension of the Ising spin lattice
 * @param lattice_dim_y The y/horizontal dimension of the Ising spin lattice
 * @param num_lattices The number of lattices in total
 */
__global__ void init_offsets_lattice(int* offset_lattice,
                                     int lattice_dim_x,
                                     int lattice_dim_y,
                                     std::size_t num_lattices);

/**
 * @brief All walkers perform fixed number of Wang Landau iterations and additionally store
 * lattice configurations for write out.
 *
 * @param d_energy Device array containing energy per lattice
 * @param d_flag_found_interval Device array indicating whether a lattice was already found for the interval
 * @param d_lattice Device array containing spin configurations
 * @param d_found_interval_lattice Device array containing spin lattices which will be written to disk
 * @param d_histogram Device array containing the histogram over the energy range
 * @param d_seed_offset Device array containing the seed offsets
 * @param d_interactions Device array containing the interactions
 * @param d_offset_lattice_per_walker Device array containing the lattice offsets for each walker
 * @param E_min Minimal reachable energy
 * @param E_max Maximal reachable energy
 * @param num_wl_iterations Number of WL iterations to perform
 * @param dimX X Dimension of a single spin lattice
 * @param dimY Y Dimension of a single spin lattice
 * @param seed Random number seed
 * @param num_intervals Number of intervals to write out
 * @param len_interval Length of a single interval. Note: All intervals are of same length, except the last one might be
 * longer.
 * @param num_total_walker Number of total WL walker
 * @param walker_per_interactions Number of WL walker per interaction
 * @param flag_found_all_intervals Flag indicating whether all intervals were found
 * @param boundary Boundary conditions, 0 -> periodic, 1 -> open. Defaults to periodic.
 */
__global__ void wang_landau_pre_run(int* d_energy,
                                    int* d_flag_found_interval,
                                    int8_t* d_lattice,
                                    int8_t* d_found_interval_lattice,
                                    unsigned long long* d_histogram,
                                    unsigned long* d_seed_offset,
                                    const signed char* d_interactions,
                                    const int* d_offset_lattice_per_walker,
                                    int E_min,
                                    int E_max,
                                    int num_wl_iterations,
                                    int dimX,
                                    int dimY,
                                    int seed,
                                    unsigned int num_intervals,
                                    int len_interval,
                                    std::size_t num_total_walker,
                                    int walker_per_interactions,
                                    wl::boundary boundary = wl::boundary::periodic);

/**
 * @brief This kernel checks whether the calculated energies for the initialized lattices
 * fit into the interval energies. If not an assert error is thrown.
 *
 * @note This kernel is called with total interval blocks and walker per intervals threads.
 *
 * @param d_flag_check_energies Device array used to indicate whether energies are in correct interval
 * @param d_energy_per_walker Lattice energies per walker
 * @param d_interval_start_energies Start energies for each interval over all interactions
 * @param d_interval_end_energies End energies for each interval over all interactions
 * @param num_total_walker Number of all walker used in the Wang Landau run
 */
__global__ void check_energy_ranges(int8_t* d_flag_check_energies,
                                    const int* d_energy_per_walker,
                                    const int* d_interval_start_energies,
                                    const int* d_interval_end_energies,
                                    std::size_t num_total_walker);

/**
 * @brief This function initializes the indices used for the replica exchange kernel.
 * Each thread assigns its threadIdx to d_replica_indices[tid].
 *
 * @note Should be called with total_interval blocks and walker_per_interval threads.
 *
 * @param d_replica_indices Device array to store the indices used to swap replicas
 * @param num_total_walker Number of all walkers used in WL
 */
__global__ void init_indices(int* d_replica_indices, std::size_t num_total_walker);

/**
 * Reset flag for finished walkers
 *
 * @note Should be called with num_interactions blocks and num_intervals threads
 *
 * @param d_cond_interval Array with the flags for each interval
 * @param d_factor The wang-landau factors
 * @param total_intervals Total number of energy intervals
 * @param beta The convergence criterion e^beta
 * @param walker_per_interval Number of walkers per interval
 */
__global__ void reset_d_cond(int8_t* d_cond_interval,
                             const double* d_factor,
                             int total_intervals,
                             double beta,
                             int walker_per_interval);

/**
 * Initialize histogram offsets for each walker
 *
 * @note Should be called with total_interval blocks and walker per interval threads.
 *
 * @param d_offset_histogram Array for histogram offsets per walker
 * @param d_interval_start_energies Array containing start energies per interval
 * @param d_interval_end_energies Array containing end energies per interval
 * @param d_len_histograms Array containing length of histograms per interaction
 * @param num_intervals Number of intervals per interaction
 * @param total_walker Number of all WL walker
 */
__global__ void init_offsets_histogram(int* d_offset_histogram,
                                       const int* d_interval_start_energies,
                                       const int* d_interval_end_energies,
                                       const int* d_len_histograms,
                                       int num_intervals,
                                       int total_walker);

/**
 * Compute average log(g(E)) within each interval.
 *
 * Number of block and threads as many as len_histogram_over_all_walkers
 *
 * @param d_shared_logG Output buffer for the avergages
 * @param d_len_histograms Array containing length of histograms per interaction
 * @param d_log_G log(g(E)) values for each walker
 * @param d_interval_start_energies Array containing start energies per interval
 * @param d_interval_end_energies Array containing end energies per interval
 * @param d_expected_energy_spectrum Array with flags for energy values
 * @param d_cond Array with flag for interactions
 * @param d_offset_histogram Array for histogram offsets per walker
 * @param d_offset_energy_spectrum Array with offsets for energy spectrum flags
 * @param d_offset_shared_logG Array with offsets for each average log(g(E))
 * @param d_cond_interaction Array with flag for interactions
 * @param num_interactions Number of interactions
 * @param num_walker_per_interval Number of walkers per intervale
 * @param num_intervals_per_interaction Number of intervals per interaction
 * @param total_len_histogram Total length of all histograms
 */
__global__ void calc_average_log_g(double* d_shared_logG,
                                   const int* d_len_histograms,
                                   const double* d_log_G,
                                   const int* d_interval_start_energies,
                                   const int* d_interval_end_energies,
                                   const int8_t* d_expected_energy_spectrum,
                                   const int8_t* d_cond,
                                   const int* d_offset_histogram,
                                   const int* d_offset_energy_spectrum,
                                   const long long* d_offset_shared_logG,
                                   const int* d_cond_interaction,
                                   int num_interactions,
                                   int num_walker_per_interval,
                                   int num_intervals_per_interaction,
                                   int total_len_histogram);

/**
 * @brief Checks whether all intervals in an interaction are finished. If yes set d_cond_interaction to 1.
 *
 * @param d_cond_interactions Array with flags for each interaction
 * @param d_tmp_storage Temporary storage used for cub device reduce
 * @param d_cond_interval Array with flags for each interval
 * @param d_offset_intervals Array with 0, num_intervals, 2*num_intervals, ..., total_intervals. Used for summing.
 * @param num_intervals Number of intervals per interaction
 * @param num_interactions Number of interactions
 */
void check_interactions_finished(int* d_cond_interactions,
                                 wl::device_tmp<void>& d_tmp_storage,
                                 const int8_t* d_cond_interval,
                                 const int* d_offset_intervals,
                                 int num_intervals,
                                 int num_interactions);

/**
 * Performs replica exchange algorithm with succeeding block. Needs to be called twice. First with all even blocks and
 * after that with all uneven blocks. Replica exchange happens by swapping lattice offsets.
 *
 * @note Should be called with total interval blocks and walker per interval threads.
 *
 * @param d_offset_lattice Array containing offsets per lattice.
 * @param d_energy Array containing energies per walker
 * @param d_replica_indices Array containing threadIdxs indices for each walker
 * @param d_seed_offset Array containing seed offsets
 * @param d_interval_start_energies Array containing interval start energies
 * @param d_interval_end_energies Array containing interval end energies
 * @param d_offset_histogram Array containing histogram offsets per walker
 * @param d_cond_interaction Array with flags for interactions
 * @param d_logG Array with log_g values per walker
 * @param even Flag to indicate whether even blocks perform replica exchange
 * @param seed random seed
 * @param num_intervals Number of intervals per interaction
 * @param walker_per_interactions Number of walkers per interaction
 */
__global__ void replica_exchange(int* d_offset_lattice,
                                 int* d_energy,
                                 int* d_replica_indices,
                                 unsigned long* d_seed_offset,
                                 const int* d_interval_start_energies,
                                 const int* d_interval_end_energies,
                                 const int* d_offset_histogram,
                                 const int* d_cond_interaction,
                                 const double* d_logG,
                                 bool even,
                                 int seed,
                                 int num_intervals,
                                 int walker_per_interactions);

/**
 * Checks whether the histogram over the current interval is thread per walker. If all walkers achieved a flat
 histogram, the WL factor is updated.
 *
 * @param d_histogram Array containing the histograms for each walker
 * @param d_factor Array containing the WL factors for each walker
 * @param d_cond_interval Array with flags for each interval
 * @param d_offset_histogram Array with histogram offsets per walker
 * @param d_interval_end_energies Array containing interval end energies
 * @param d_interval_start_energies Array containing interval start energies
 * @param d_expected_energy_spectrum Array with flags for energy values
 * @param d_offset_energy_spectrum Array with offsets for energy spectrum flags
 * @param d_cond_interaction Array with flag for interactions
 * @param alpha alpha used in WL
 * @param num_walker_total Number of all walkers used
 * @param walker_per_interactions Walker per interactions
 * @param num_intervals Num_intervals per interaction

 */
__global__ void check_histogram(unsigned long long* d_histogram,
                                double* d_factor,
                                int8_t* d_cond_interval,
                                const int* d_offset_histogram,
                                const int* d_interval_end_energies,
                                const int* d_interval_start_energies,
                                const int8_t* d_expected_energy_spectrum,
                                const int* d_offset_energy_spectrum,
                                const int* d_cond_interaction,
                                double alpha,
                                int num_walker_total,
                                int walker_per_interactions,
                                int num_intervals);

/**
 * Redistributes average log_g values back to the log_g values per walker.
 *
 * @param d_log_G Array containing log_g values per walker
 * @param d_shared_logG Array containing averaged log_g values over walker
 * @param d_len_histograms Array containing length of histograms over walker
 * @param d_interval_start_energies Array containing interval start energies
 * @param d_interval_end_energies Array containing interval end energies
 * @param d_cond_interaction Array with flags for interaction
 * @param d_offset_histogram Array with histogram offsets per walker
 * @param d_cond_interval Array with flags for intervals
 * @param d_offset_shared_logG Array with offsets for shared log_gs
 * @param num_intervals_per_interaction Num intervals per interaction
 * @param num_walker_per_interval Num walker per interval
 * @param beta Beta from WL algorithm
 * @param num_interactions Number of interactions
 * @param total_len_histogram Total length of histogram
 */

__global__ void redistribute_g_values(double* d_log_G,
                                      const double* d_shared_logG,
                                      const int* d_len_histograms,
                                      const int* d_interval_start_energies,
                                      const int* d_interval_end_energies,
                                      const int* d_cond_interaction,
                                      const int* d_offset_histogram,
                                      const int8_t* d_cond_interval,
                                      const long long* d_offset_shared_logG,
                                      int num_intervals_per_interaction,
                                      int num_walker_per_interval,
                                      int num_interactions,
                                      int total_len_histogram);

/**
 * @brief Replica Exchange Wang-Landau algorithm kernel for 2D RBIM.
 *
 * Each thread simulates one walker for a specific interaction configuration.
 * The goal is to perform random walk in energy space and update the density of states (`logG`)
 * and histogram (`H`) using the Wang-Landau algorithm, or detect if an unexpected energy is encountered.
 *
 * @param d_lattice Device pointer to lattice spin configurations.
 * @param d_interactions Device pointer to interaction configurations (couplings).
 * @param d_H Device pointer to histogram of visited energies.
 * @param d_logG Device pointer to estimated log density of states.
 * @param factor Device pointer to multiplicative Wang-Landau factor per thread.
 * @param d_energy Device pointer to current energy per walker.
 * @param d_newEnergies Device pointer to store newly found (unexpected) energy values.
 * @param d_offset_iter Device pointer to iteration counters per walker.
 * @param foundFlag Device pointer to flag unexpected energy hits per walker.
 * @param d_start Device pointer to interval start energies per interaction.
 * @param d_end Device pointer to interval end energies per interaction.
 * @param d_offset_histogram Device pointer to histogram index offset per walker.
 * @param d_offset_lattice Device pointer to lattice index offset per walker.
 * @param d_cond Device pointer indicating sampling condition (0: WL update, 1: no update).
 * @param d_expected_energy_spectrum Pointer to expected energy spectrum (binary mask).
 * @param d_offset_energy_spectrum Pointer to energy spectrum index offset per interaction.
 * @param d_cond_interaction Device pointer indicating if interaction is skipped (-1: skip).
 * @param dimX Width of the lattice.
 * @param dimY Height of the lattice.
 * @param num_iterations Number of Monte Carlo steps to perform.
 * @param num_lattices Total number of walkers (threads).
 * @param num_intervals Number of energy intervals per interaction.
 * @param walker_per_interactions Number of walkers per interaction configuration.
 * @param seed Random seed.
 * @param boundary Boundary condition identifier (periodic, open).
 */
__global__ void wang_landau(signed char* d_lattice,
                            signed char* d_interactions,
                            unsigned long long* d_H,
                            double* d_logG,
                            double* factor,
                            int* d_energy,
                            int* d_newEnergies,
                            unsigned long* d_offset_iter,
                            int* foundFlag,
                            const int* d_start,
                            const int* d_end,
                            const int* d_offset_histogram,
                            const int* d_offset_lattice,
                            const int8_t* d_cond,
                            const int8_t* d_expected_energy_spectrum,
                            const int* d_offset_energy_spectrum,
                            const int* d_cond_interaction,
                            int dimX,
                            int dimY,
                            int num_iterations,
                            int num_lattices,
                            int num_intervals,
                            int walker_per_interactions,
                            int seed,
                            wl::boundary boundary = wl::boundary::periodic);

/**
* Performs the fisher yates algorithm on the d_shuffle array, i.e. it shuffles the thread indices in order to determine
* with which thread the replica exchange is performed. Fisher yates
* (https://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle)
*
* @param d_shuffle Array containing threadIdx indices
* @param d_seed_offset Array containing seed offsets
* @param seed Random number seed
* @param interaction_id Interaction id
* @param walker_per_interactions WL Walker per interactions
*/
__device__ void fisher_yates(int* d_shuffle,
                             unsigned long* d_seed_offset,
                             int seed,
                             int interaction_id,
                             int walker_per_interactions);
} // wl

#endif // WL_CUH_
