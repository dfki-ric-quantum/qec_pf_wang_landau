#ifndef PROGRAM_OPTIONS_HPP_
#define PROGRAM_OPTIONS_HPP_

#include <tuple>

#include "boundary.hpp"

namespace wl {

/**
 * Program options for the WL pre-run
 */
struct prerun_options
{
  int X{};                       /**< Ising lattice horizontal dimension */
  int Y{};                       /**< Ising lattice vertical dimension */
  float prob_interactions{};     /**< bit flip error probability */
  int num_iterations{};          /**< loop iteration count */
  int walker_per_interactions{}; /**< number of walker per disorder sample */
  int seed{};                    /**< start seed used for disorder sampling */
  int task_id{};                 /**< task specifier for simultaneous runs */
  unsigned int num_intervals{};  /**< number of intervals to cover energy spectrum with */
  char logical_error_type = 'i'; /**< specififer for logical error coset representative - default trivial */
  int num_interactions{};        /**< number of disorder samples */
  wl::boundary boundary{};       /**< Boundary condition 0 = periodic, 1 = open */
  bool skip_output = false;      /**< Skip writing results to the filesystem */
};

/**
 * Program options for the WL main run
 */
struct mainrun_options
{
  int X{};                       /**< Ising spin lattice vertical dimension */
  int Y{};                       /**< Ising spin lattice horizontal dimension */
  int num_iterations{};          /**< Number of Wang Landau steps per check flatness iteration */
  float prob_interactions{};     /**< Physical probability rate of bit flip error on qubit */
  double alpha{};                /**< Flatness parameter */
  double beta{};                 /**< Depth parameter for Wang Landau factor updates */
  int num_intervals{};           /**< Number of intervals to cover energy specturm with */
  int walker_per_interval{};     /**< Number of independent walkers per interval */
  float overlap_decimal{};       /**< Overlap of neighboring intervals as decimal number */
  int seed{};                    /**< Seed for Wang Landau steps */
  char logical_error_type{};     /**< Error class specifier */
  int num_interactions{};        /**< Number of indpenedent disorder samples to simulate */
  wl::boundary boundary{};       /**< Boundary condition 0 = periodic, 1 = open */
  int replica_exchange_offset{}; /**< Number of check flatness checks before replica exchange is executed */
  int task_id{};                 /**< Task id to seperate parallel Wang Landau jobs for parallel execution on cluster */
  int time_limit{};              /**<Time limit to kill Wang Landau algorithm execution */
};

/** Template type for option parsing results */
template<typename Opt>
using opt_parse_res = std::tuple<Opt, bool>;

using prerun_opt_res = opt_parse_res<prerun_options>;
using mainrun_opt_res = opt_parse_res<mainrun_options>;

/**
 * Parse options for the WL pre run
 *
 * @param argc number of arguments, as in main()
 * @param argv the arguments, as in main()
 * @return The parsed options and `true` on success, default initialized options and false
 * otherwise.
 */
[[nodiscard]] auto parse_prerun_options(int argc, char** argv) -> prerun_opt_res;

/**
 * Parse options for the WL main run
 *
 * @param argc number of arguments, as in main()
 * @param argv the arguments, as in main()
 * @return The parsed options and `true` on success, default initialized options and false
 * otherwise.
 */
[[nodiscard]] auto parse_mainrun_options(int argc, char** argv) -> mainrun_opt_res;

} // wl

#endif // PROGRAM_OPTIONS_HPP_
