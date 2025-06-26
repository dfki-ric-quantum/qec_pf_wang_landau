#include <H5Apublic.h>
#include <H5Dpublic.h>
#include <H5Spublic.h>
#include <boost/range/combine.hpp>

#include <chrono>
#include <ctime>
#include <filesystem>
#include <format>
#include <hdf5.h>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include "boundary.hpp"
#include "program_options.hpp"
#include "utils.hpp"

namespace {
hid_t create_or_open_group(hid_t parent_group_id, const std::string_view group_name)
{
  if (H5Lexists(parent_group_id, group_name.data(), H5P_DEFAULT) > 0) {
    hid_t group_id = H5Gopen2(parent_group_id, group_name.data(), H5P_DEFAULT);
    if (group_id < 0) {
      throw std::runtime_error(std::format("Error: Could not open existing group {}", group_name));
    }
    return group_id;
  } else {
    hid_t group_id = H5Gcreate2(parent_group_id, group_name.data(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (group_id < 0) {
      throw std::runtime_error(std::format("Error: Could not create group {}", group_name));
    }
    return group_id;
  }
}

struct ResultEntry
{
  int key;
  double value;
};
} // namespace

void wl::log(std::string_view msg, log_level level)
{
  const auto now = std::chrono::system_clock::now();
  const auto time = std::chrono::system_clock::to_time_t(now);

  // clang-format off
  const char levelc = [level] () {
    switch (level) {
      case log_level::info:
        return 'I';
      case log_level::warn:
        return 'W';
      case log_level::error:
        return 'E';
      default:
        return 'U';
    }
  }();

  std::clog << levelc << ' '
            << std::put_time(std::localtime(&time), "%H:%M:%S - %d/%m/%Y")
            << " -- " << msg << '\n';
  // clang-format on
}

void wl::log_intervals(const wl::interval_results& ivr)
{
  log("Intervals for the run");

  for (const auto& zipped : boost::combine(ivr.h_start, ivr.h_end)) {
    int start{};
    int end{};
    boost::tie(start, end) = zipped;
    log(std::format("{} {}", start, end));
  }
}

int wl::get_seed_from_os()
{
  std::random_device rand_dev;
  return static_cast<int>(rand_dev());
}

auto wl::build_prerun_path(const prerun_options& options) -> std::string
{
  const int precision = 6;

  std::stringstream strstr;
  strstr << "init/task_id_" << options.task_id << "/seed_" << options.seed;
  strstr << "/" << wl::boundary_to_str(options.boundary);
  strstr << "/prob_" << std::fixed << std::setprecision(precision) << options.prob_interactions;
  strstr << "/X_" << options.X << "_Y_" << options.Y;
  strstr << "/error_class_" << options.logical_error_type;

  return strstr.str();
}

std::string wl::build_mainrun_path(const mainrun_options& options)
{
  const int precision = 6;

  std::stringstream strstr;
  strstr << "init/task_id_" << options.task_id << "/seed_" << options.seed;
  strstr << "/" << wl::boundary_to_str(options.boundary);
  strstr << "/prob_" << std::fixed << std::setprecision(precision) << options.prob_interactions;
  strstr << "/X_" << options.X << "_Y_" << options.Y;
  strstr << "/error_class_" << options.logical_error_type;

  return strstr.str();
}

auto wl::build_mainrun_result_path(const mainrun_options& options) -> std::string
{
  const int precision = 6;

  std::stringstream strstr;
  strstr << "results/task_id_" << options.task_id << "/seed_" << options.seed;
  strstr << "/" << wl::boundary_to_str(options.boundary);
  strstr << "/prob_" << std::fixed << std::setprecision(precision) << options.prob_interactions;
  strstr << "/X_" << options.X << "_Y_" << options.Y;
  strstr << "/error_class_" << options.logical_error_type;

  return strstr.str();
}

void wl::create_directories(std::string_view path)
{
  if (!std::filesystem::exists(path)) {
    std::filesystem::create_directories(path);
  }
}

void wl::write_results(const std::vector<std::map<int, double>>& result_data,
                       const mainrun_options& options,
                       int disorder_id,
                       const std::string_view file_path,
                       const std::string_view timestamp_group_name,
                       const std::string_view git_info)
{
  // Flatten the result_data into a single vector
  std::vector<ResultEntry> flattened_data;
  for (const auto& map : result_data) {
    for (const auto& [key, value] : map) {
      flattened_data.push_back({key, value});
    }
  }

  std::string result_path = std::string(file_path) + "/mainrun_results.h5";

  // Create or open the HDF5 file
  hid_t file_id = -1;
  if (!std::filesystem::exists(result_path)) {
    file_id = H5Fcreate(result_path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (file_id < 0) {
      throw std::runtime_error{"Error: Could not create result file.\n"};
    }
  } else {
    file_id = H5Fopen(result_path.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
    if (file_id < 0) {
      throw std::runtime_error{"Error: Could not read existing result file.\n"};
    }
  }

  try {
    std::string disorder_group_name = std::format("/{}", disorder_id);
    hid_t disorder_group_id = create_or_open_group(file_id, disorder_group_name);

    std::string run_group_name = std::format("{}", options.seed);
    hid_t run_group_id = create_or_open_group(disorder_group_id, run_group_name);

    hid_t timestamp_group_id = create_or_open_group(run_group_id, timestamp_group_name.data());

    // here comes write dataset and code version as attribute
    hid_t compound_type = H5Tcreate(H5T_COMPOUND, sizeof(ResultEntry));
    H5Tinsert(compound_type, "Key", HOFFSET(ResultEntry, key), H5T_NATIVE_INT);
    H5Tinsert(compound_type, "Value", HOFFSET(ResultEntry, value), H5T_NATIVE_DOUBLE);

    hsize_t num_entries = flattened_data.size();
    hsize_t dims[1] = {num_entries}; // 1D array of compund result types
    hid_t dataspace_id = H5Screate_simple(1, dims, nullptr);

    hid_t dataset_id =
      H5Dcreate(timestamp_group_id, "log_g", compound_type, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    if (dataset_id < 0) {
      throw std::runtime_error{"Error: Could not create log_g dataset in run group.\n"};
    }

    if (H5Dwrite(dataset_id, compound_type, H5S_ALL, H5S_ALL, H5P_DEFAULT, flattened_data.data()) < 0) {
      throw std::runtime_error{"Error: Could not write data to dataset.\n"};
    }

    // Add the code version as a string type of variable length as attribute to dataset
    hid_t attr_dataspace_id = H5Screate(H5S_SCALAR);
    if (attr_dataspace_id < 0) {
      throw std::runtime_error{"Error: Could not create attribute dataspace.\n"};
    }
    // Define the datatype for max 256 length string
    hid_t str_type = H5Tcopy(H5T_C_S1);
    H5Tset_size(str_type, 256);
    H5Tset_strpad(str_type, H5T_STR_NULLTERM); // Ensure NULL termination for shorter strings

    const char* git_info_c_str = git_info.data();

    hid_t attr_id = H5Acreate(dataset_id, "CodeVersion", str_type, attr_dataspace_id, H5P_DEFAULT, H5P_DEFAULT);
    if (attr_id < 0) {
      throw std::runtime_error{"Error: Could not create attribute for dataset.\n"};
    }
    if (H5Awrite(attr_id, str_type, git_info_c_str) < 0) {
      throw std::runtime_error{"Error: Could not write attribute to dataset.\n"};
      H5Aclose(attr_id);
      wl::log("attr id");
      H5Tclose(str_type);
      wl::log("str type");
      H5Sclose(attr_dataspace_id);
      wl::log("attr dataspace id");
    }

    H5Aclose(attr_id);
    H5Tclose(str_type);
    H5Sclose(attr_dataspace_id);
    H5Dclose(dataset_id);
    H5Sclose(dataspace_id);
    H5Gclose(timestamp_group_id);
    H5Gclose(run_group_id);
    H5Gclose(disorder_group_id);
  } catch (...) {
    H5Fclose(file_id);
    throw;
  }
  H5Fclose(file_id);
}

std::string wl::get_timestamp_group_name()
{
  const auto now = std::chrono::system_clock::now();
  const auto time = std::chrono::system_clock::to_time_t(now);
  std::string timestamp_id = ctime(&time);
  if (!timestamp_id.empty() && timestamp_id.back() == '\n') {
    timestamp_id.pop_back();
  }
  std::string timestamp_group_name = std::format("{}", timestamp_id);
  std::replace(timestamp_group_name.begin(), timestamp_group_name.end(), ' ', '_');

  return timestamp_group_name;
}

int wl::cast_safely_longlong_to_int(long long value)
{
  const long long int_min = std::numeric_limits<int>::min();
  const long long int_max = std::numeric_limits<int>::max();

  if (value < int_min || value > int_max) {
    throw std::out_of_range("Value cannot be safely cast to int: out of range.");
  }

  return static_cast<int>(value);
}
