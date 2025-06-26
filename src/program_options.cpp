#include <boost/program_options.hpp>
#include <iostream>
#include <string>
#include <string_view>

#include "boundary.hpp"
#include "program_options.hpp"
#include "utils.hpp"

namespace po = boost::program_options;

namespace {
inline void _print_help(std::string_view prog_name, const po::options_description& desc)
{
  std::cout << prog_name << "\n\n" << desc << '\n';
}

inline wl::boundary _map_boundary(unsigned int boundary) {
  switch (boundary) {
    case 0:
      return wl::boundary::periodic;
    case 1:
      return wl::boundary::open;
    default:
      throw std::runtime_error(std::format("Unsupported boundary condition {}", boundary));
  }
}
} //

auto wl::parse_prerun_options(int argc, char** argv) -> prerun_opt_res
{
  prerun_options options;
  unsigned int boundary = 0;

  po::options_description desc("WL pre-run usage");
  // clang-format off
  desc.add_options()
      ("help,h", "Help")
      ("X,x",
        po::value<int>(&options.X)->required(), "Ising lattice, horizontal dimension")
      ("Y,y",
        po::value<int>(&options.Y)->required(), "Ising lattice, vertical dimension")
      ("prob,p",
        po::value<float>(&options.prob_interactions),
        "Bit-flip error probablities")
      ("nit,n",
        po::value<int>(&options.num_iterations)->required(),
        "Number of WL iterations")
      ("nw,w",
        po::value<int>(&options.walker_per_interactions)->required(),
        "Number of walkers per disorder sample")
      ("seed,s",
        po::value<int>(&options.seed),
        "Random seed")
      ("num-intervals,i",
        po::value<unsigned int>(&options.num_intervals)->required(),
        "Number of intervals in the energy spectrum")
      ("logical-error,e",
        po::value<char>(&options.logical_error_type)->default_value('i'),
        "type of logical error coset representative")
      ("disorder-samples,r",
        po::value<int>(&options.num_interactions)->required(),
        "Number of disorder samples")
      ("boundary",
       po::value<unsigned int>(&boundary)->default_value(0),
       "Boundary condition")
      ("task_id,d",
        po::value<int>(&options.task_id)->required(),
        "Task ID")
      ("skip-output", "Skip writing results to the filesystem")
      ("version,v", "Show version");
  // clang-format on

  po::variables_map variables_map;
  po::store(po::parse_command_line(argc, argv, desc), variables_map);

  if (variables_map.count("help") > 0) {
    _print_help(argv[0], desc);
    return {options, false};
  }

  if (variables_map.count("version") > 0) {
    std::cout << "WL prerun version: " << wl::get_git_version() << '\n';
    return {options, false};
  }

  try {
    po::notify(variables_map);
    options.skip_output = static_cast<bool>(variables_map.count("skip-output"));

    if (variables_map.count("seed") == 0) {
      options.seed = wl::get_seed_from_os();
    }
    options.boundary = _map_boundary(boundary);
  } catch (po::required_option& err) {
    std::cerr << err.what() << '\n';
    _print_help(argv[0], desc);
    return {options, false};
  }
  return {options, true};
}

auto wl::parse_mainrun_options(int argc, char** argv) -> mainrun_opt_res
{
  mainrun_options options;
  unsigned int boundary = 0;

  po::options_description desc("WL main usage");
  // clang-format off
  desc.add_options()
      ("help", "Help")
      ("X,x",
        po::value<int>(&options.X)->required(), "X")
      ("Y,y",
        po::value<int>(&options.Y)->required(), "Y")
      ("num_iterations,n",
        po::value<int>(&options.num_iterations)->required(),
        "Number of iterations")
      ("prob_interactions,p",
         po::value<float>(&options.prob_interactions)->required(),
         "Interaction probabilities")
      ("alpha,a",
         po::value<double>(&options.alpha)->required(),
         "Alpha")
      ("beta,b",
         po::value<double>(&options.beta)->required(),
         "Beta")
      ("num_intervals,i",
        po::value<int>(&options.num_intervals)->required(),
        "Number of intervals")
      ("walker_per_interval,w",
         po::value<int>(&options.walker_per_interval)->required(),
         "Number of walkers per interval")
      ("overlap_decimal,o",
         po::value<float>(&options.overlap_decimal)->required(),
         "Number of walkers per interval")
      ("seed,s",
        po::value<int>(&options.seed),
        "Random seed (Run)")
      ("logical_error,e",
        po::value<char>(&options.logical_error_type)->default_value('I'),
        "Type of logical error")
      ("repetitions_interactions,r",
        po::value<int>(&options.num_interactions)->required(),
        "Repetitions interactions")
      ("replica_exchange_offsets,c",
        po::value<int>(&options.replica_exchange_offset)->required(),
        "Repetitions interactions")
      ("boundary",
       po::value<unsigned int>(&boundary)->default_value(0),
       "Boundary condition")
      ("task_id,d",
       po::value<int>(&options.task_id)->required(),
       "Task ID")
      ("time_limit,f", po::value<int>(&options.time_limit)->required(),
       "Time Limit")
      ("version,v", "Show version");
  // clang-format on

  po::variables_map variables_map;
  po::store(po::parse_command_line(argc, argv, desc), variables_map);

  if (variables_map.count("help") > 0) {
    _print_help(argv[0], desc);
    return {options, false};
  }

  if (variables_map.count("version") > 0) {
    std::cout << "WL mainrun version: " << wl::get_git_version() << '\n';
    return {options, false};
  }

  try {
    po::notify(variables_map);

    if (variables_map.count("seed") == 0) {
      options.seed = wl::get_seed_from_os();
    }
    options.boundary = _map_boundary(boundary);
  } catch (po::required_option& err) {
    std::cerr << err.what() << '\n';
    _print_help(argv[0], desc);
    return {options, false};
  }
  return {options, true};
}
