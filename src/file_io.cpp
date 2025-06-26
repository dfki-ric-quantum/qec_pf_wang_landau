#include <hdf5.h>

#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <format>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>

#include "file_io.hpp"
#include "program_options.hpp"

hid_t check_hdf5_status(hid_t status_id, const std::string_view message)
{
  if (status_id < 0) {
    throw std::runtime_error{std::format("{}\n", message)};
  }
  return status_id;
}

void wl::write_prerun_histogram_results(const std::vector<unsigned long long>& h_histograms,
                                        const wl::prerun_options& options,
                                        std::size_t len_histogram,
                                        hid_t file_id)
{
  // Temporary buffer to store transformed histogram data (0 or 1 only)
  std::vector<unsigned char> binary_histogram(len_histogram * static_cast<std::size_t>(options.num_interactions));

  for (std::size_t i = 0; i < static_cast<std::size_t>(options.num_interactions); ++i) {
    // Transform the corresponding portion of h_histograms to binary
    for (std::size_t j = 0; j < len_histogram; ++j) {
      binary_histogram[i * len_histogram + j] = h_histograms[i * len_histogram + j] != 0 ? 1 : 0;
    }
  }

  if (H5Lexists(file_id, "Histogram", H5P_DEFAULT) > 0) {
    check_hdf5_status(H5Ldelete(file_id, "Histogram", H5P_DEFAULT), "Failed to delete existing Histogram group");
  }

  hid_t histogram_group_id = check_hdf5_status(H5Gcreate2(file_id, "Histogram", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT),
                                               "Failed to create Histogram group");

  for (int i = 0; i < options.num_interactions; ++i) {
    std::string disorder_sample_offset = std::to_string(i);

    hid_t group_id = check_hdf5_status(
      H5Gcreate2(histogram_group_id, disorder_sample_offset.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT),
      "Failed to create h5 group");

    // Write histogram data
    hsize_t histogram_dims[1];
    histogram_dims[0] = static_cast<hsize_t>(len_histogram);

    hid_t histogram_dataspace_id =
      check_hdf5_status(H5Screate_simple(1, histogram_dims, NULL), "Failed to create h5 dataspace");

    hid_t histogram_dataset_id = check_hdf5_status(
      H5Dcreate(
        group_id, "Histogram", H5T_NATIVE_ULLONG, histogram_dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT),
      "Failed to create h5 dataset.");

    H5Dwrite(histogram_dataset_id,
             H5T_NATIVE_UCHAR,
             H5S_ALL,
             H5S_ALL,
             H5P_DEFAULT,
             binary_histogram.data() + static_cast<std::size_t>(i) * len_histogram);

    H5Sclose(histogram_dataspace_id);
    H5Dclose(histogram_dataset_id);
    H5Gclose(group_id);
  }
  H5Gclose(histogram_group_id);
}

void wl::write_prerun_interaction_results(const std::vector<signed char>& h_interactions,
                                          const wl::prerun_options& options,
                                          hid_t file_id)
{
  int len_interaction = 2 * options.X * options.Y;

  if (H5Lexists(file_id, "Interaction", H5P_DEFAULT) > 0) {
    check_hdf5_status(H5Ldelete(file_id, "Interaction", H5P_DEFAULT), "Failed to delete existing Interaction group");
  }

  hid_t interaction_group_id = check_hdf5_status(
    H5Gcreate2(file_id, "Interaction", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), "Failed to create Interaction group");

  for (int i = 0; i < options.num_interactions; ++i) {
    // Create a group for each disorder sample
    // identifier is seed used to generate the disorder sample which  has to be consistent
    // with random number generation in init interaction
    std::string disorder_sample = std::to_string(i);

    hid_t group_id = check_hdf5_status(
      H5Gcreate2(interaction_group_id, disorder_sample.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT),
      "Failed to create h5 group");

    // Write interaction data for disorder sample
    hsize_t interaction_dims[1];
    interaction_dims[0] = static_cast<hsize_t>(len_interaction);

    hid_t interaction_dataspace_id =
      check_hdf5_status(H5Screate_simple(1, interaction_dims, NULL), "Failed to create h5 dataspace");

    hid_t interaction_dataset_id = check_hdf5_status(
      H5Dcreate(
        group_id, "Interaction", H5T_NATIVE_SCHAR, interaction_dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT),
      "Failed to create h5 dataset.");

    H5Dwrite(interaction_dataset_id,
             H5T_NATIVE_SCHAR,
             H5S_ALL,
             H5S_ALL,
             H5P_DEFAULT,
             h_interactions.data() + i * len_interaction);

    H5Sclose(interaction_dataspace_id);
    H5Dclose(interaction_dataset_id);
    H5Gclose(group_id);
  }
  H5Gclose(interaction_group_id);
}

void wl::write_prerun_lattice_results(const std::vector<signed char>& h_lattices,
                                      const std::vector<int>& h_energies,
                                      const wl::prerun_options& options,
                                      hid_t file_id)
{
  auto len_lattice = std::size_t(options.X * options.Y);
  auto num_intervals = static_cast<std::size_t>(options.num_intervals);

  if (H5Lexists(file_id, "Lattice", H5P_DEFAULT) > 0) {
    check_hdf5_status(H5Ldelete(file_id, "Lattice", H5P_DEFAULT), "Failed to delete existing Histogram group");
  }

  hid_t lattice_group_id = check_hdf5_status(H5Gcreate2(file_id, "Lattice", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT),
                                             "Failed to create Histogram group");

  for (int i = 0; i < options.num_interactions; ++i) {
    std::string disorder_sample = std::to_string(i);

    hid_t disorder_group_id =
      check_hdf5_status(H5Gcreate2(lattice_group_id, disorder_sample.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT),
                        "Failed to create disorder group for prerun lattice result");

    for (std::size_t j = 0; j < num_intervals; ++j) {
      // check if spin config for this interval was found
      if (h_lattices[len_lattice * (num_intervals * static_cast<std::size_t>(i) + j)] != 0) {
        // Create a group for each occupied energy interval - nested group follows
        std::ostringstream energy;
        energy << std::to_string(h_energies[num_intervals * static_cast<std::size_t>(i) + j]);

        hid_t interval_group_id =
          check_hdf5_status(H5Gcreate2(disorder_group_id, energy.str().c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT),
                            "Failed to create h5 interval group for lattice write");

        // Write interaction data for disorder sample
        hsize_t lattice_dims[1];
        lattice_dims[0] = static_cast<hsize_t>(len_lattice);

        hid_t lattice_dataspace_id =
          check_hdf5_status(H5Screate_simple(1, lattice_dims, NULL), "Failed to create h5 dataspace");

        hid_t lattice_dataset_id = check_hdf5_status(H5Dcreate(interval_group_id,
                                                               "Lattice",
                                                               H5T_NATIVE_SCHAR,
                                                               lattice_dataspace_id,
                                                               H5P_DEFAULT,
                                                               H5P_DEFAULT,
                                                               H5P_DEFAULT),
                                                     "Failed to create h5 dataset.");

        H5Dwrite(lattice_dataset_id,
                 H5T_NATIVE_SCHAR,
                 H5S_ALL,
                 H5S_ALL,
                 H5P_DEFAULT,
                 h_lattices.data() + static_cast<std::size_t>(i) * len_lattice * options.num_intervals +
                   j * len_lattice);

        H5Sclose(lattice_dataspace_id);
        H5Dclose(lattice_dataset_id);
        H5Gclose(interval_group_id);
      }
    }
    H5Gclose(disorder_group_id);
  }
  H5Gclose(lattice_group_id);
}

void wl::write_prerun_results(const std::vector<unsigned long long>& h_histograms,
                              const std::vector<signed char>& h_interactions,
                              const std::vector<signed char>& h_lattices,
                              const std::vector<int>& h_energies,
                              const wl::prerun_options& options,
                              std::size_t len_histogram,
                              const std::string_view file_path)
{

  std::string histogram_path = std::string(file_path) + "/prerun_results.h5";

  if (std::filesystem::exists(histogram_path)) {
    throw std::runtime_error(std::format("Prerun result file {} already exists, aborting.", histogram_path));
  }

  hid_t file_id = check_hdf5_status(H5Fcreate(histogram_path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT),
                                    "Failed to create h5 file.");

  write_prerun_histogram_results(h_histograms, options, len_histogram, file_id);
  write_prerun_interaction_results(h_interactions, options, file_id);
  write_prerun_lattice_results(h_lattices, h_energies, options, file_id);

  H5Fclose(file_id);
}

void wl::read_prerun_histograms(std::map<int, std::vector<unsigned char>>& h_histograms_dest,
                                const std::string_view file_path,
                                std::size_t len_histogram)
{
  std::string histogram_path = std::string(file_path) + "/prerun_results.h5";

  hid_t faplist_id = H5Pcreate(H5P_FILE_ACCESS);
  H5Pset_fapl_core(faplist_id, 1, false);
  hid_t file_id = check_hdf5_status(H5Fopen(histogram_path.c_str(), H5F_ACC_RDONLY, faplist_id),
                                    std::format("Error: Could not open HDF5 file:{} \n", histogram_path));

  hid_t root_group =
    check_hdf5_status(H5Gopen(file_id, "/Histogram", H5P_DEFAULT), "Error: Could not open root group \n");

  hsize_t num_objs = 0;
  H5Gget_num_objs(root_group, &num_objs);

  for (hsize_t i = 0; i < num_objs; ++i) {
    const std::size_t group_name_len = 256;
    std::string group_name(group_name_len, '\0');
    ssize_t name_len = H5Gget_objname_by_idx(root_group, i, &group_name[0], group_name_len);

    if (name_len < 0) {
      throw std::runtime_error{"Error: Could not retrieve a group name.\n"};
    }

    hid_t group_id = check_hdf5_status(H5Gopen(root_group, group_name.c_str(), H5P_DEFAULT),
                                       std::format("Error: Could not open HDF5 group:{} \n", group_name));

    hid_t dataset_id =
      check_hdf5_status(H5Dopen(group_id, "Histogram", H5P_DEFAULT), "Error: Could not open HDF5 histogram dataset \n");

    hid_t dataspace_id = check_hdf5_status(H5Dget_space(dataset_id), "Could not get groups histogram dataspace");

    hid_t ndims = check_hdf5_status(H5Sget_simple_extent_ndims(dataspace_id), "Error: Dataset has no dimensions.");

    if (ndims != 1) {
      throw std::runtime_error{"number dimension of dataspace not 1.\n"};
    }

    hsize_t dims[1];
    check_hdf5_status(H5Sget_simple_extent_dims(dataspace_id, dims, NULL), "could not get dataspace size \n");

    if (dims[0] != len_histogram) {
      H5Sclose(dataspace_id);
      H5Dclose(dataset_id);
      H5Gclose(group_id);
      throw std::runtime_error{
        std::format("Histogram length {} does not coincide with length of histogram written to file {} \n",
                    len_histogram,
                    dims[0])};
    }

    std::vector<unsigned char> histogram(len_histogram);
    H5Dread(dataset_id, H5T_NATIVE_UCHAR, H5S_ALL, H5S_ALL, H5P_DEFAULT, histogram.data());
    h_histograms_dest[std::stoi(group_name)] = std::move(histogram); // avoids unnecessary copy

    H5Gclose(group_id);
    H5Dclose(dataset_id);
    H5Sclose(dataspace_id);
  }
  H5Gclose(root_group);
  H5Fclose(file_id);
}

void wl::read_prerun_interactions(std::map<int, std::vector<signed char>>& h_interactions_dest,
                                  const std::string_view file_path,
                                  const wl::prerun_options options)
{
  auto len_interaction = static_cast<std::size_t>(options.X * options.Y * 2);

  std::string interaction_path = std::string(file_path) + "/prerun_results.h5";

  hid_t faplist_id = H5Pcreate(H5P_FILE_ACCESS);
  H5Pset_fapl_core(faplist_id, 1, false);
  hid_t file_id = check_hdf5_status(H5Fopen(interaction_path.c_str(), H5F_ACC_RDONLY, faplist_id),
                                    std::format("Error: Could not open HDF5 file:{} \n", interaction_path));

  hid_t root_group_id =
    check_hdf5_status(H5Gopen(file_id, "/Interaction", H5P_DEFAULT), "Error: Could not open root group \n");

  hsize_t num_objs;
  H5Gget_num_objs(root_group_id, &num_objs);

  for (hsize_t i = 0; i < num_objs; ++i) {
    const std::size_t group_name_len = 256;
    std::string group_name(group_name_len, '\0');
    ssize_t name_len = H5Gget_objname_by_idx(root_group_id, i, &group_name[0], group_name_len);

    if (name_len < 0) {
      throw std::runtime_error{"Error: Could not retrieve a group name.\n"};
    }

    hid_t group_id = check_hdf5_status(H5Gopen(root_group_id, group_name.c_str(), H5P_DEFAULT),
                                       std::format("Error: Could not open HDF5 group:{} \n", group_name));

    hid_t dataset_id = check_hdf5_status(H5Dopen(group_id, "Interaction", H5P_DEFAULT),
                                         "Error: Could not open HDF5 interaction dataset \n");

    hid_t dataspace_id = check_hdf5_status(H5Dget_space(dataset_id), "Could not get groups interaction dataspace");

    hsize_t dims[1];
    H5Sget_simple_extent_dims(dataspace_id, dims, nullptr);

    if (dims[0] != len_interaction) {
      // Dimension mismatch, skip this dataset
      H5Sclose(dataspace_id);
      H5Dclose(dataset_id);
      H5Gclose(group_id);
      throw std::runtime_error{
        std::format("Interaction length {} does not coincide with length of interaction written to file {} \n",
                    len_interaction,
                    dims[0])};
    }

    std::vector<signed char> interaction(len_interaction);
    H5Dread(dataset_id, H5T_NATIVE_SCHAR, H5S_ALL, H5S_ALL, H5P_DEFAULT, interaction.data());
    h_interactions_dest[std::stoi(group_name)] = std::move(interaction);

    H5Sclose(dataspace_id);
    H5Dclose(dataset_id);
    H5Gclose(group_id);
  }

  // Close the root group and file
  H5Gclose(root_group_id);
  H5Fclose(file_id);
}

void wl::read_prerun_interactions(std::map<int, std::vector<signed char>>& h_interactions_dest,
                                  const std::string_view file_path,
                                  const wl::mainrun_options& options)
{
  auto len_interaction = static_cast<std::size_t>(options.X * options.Y * 2);

  std::string interaction_path = std::string(file_path) + "/prerun_results.h5";

  hid_t faplist_id = H5Pcreate(H5P_FILE_ACCESS);
  H5Pset_fapl_core(faplist_id, 1, false);
  hid_t file_id = check_hdf5_status(H5Fopen(interaction_path.c_str(), H5F_ACC_RDONLY, faplist_id),
                                    std::format("Error: Could not open HDF5 file:{} \n", interaction_path));

  hid_t root_group_id =
    check_hdf5_status(H5Gopen(file_id, "/Interaction", H5P_DEFAULT), "Error: Could not open root group \n");

  hsize_t num_objs;
  H5Gget_num_objs(root_group_id, &num_objs);

  for (hsize_t i = 0; i < num_objs; ++i) {
    const std::size_t group_name_len = 256;
    std::string group_name(group_name_len, '\0');
    ssize_t name_len = H5Gget_objname_by_idx(root_group_id, i, &group_name[0], group_name_len);

    if (name_len < 0) {
      throw std::runtime_error{"Error: Could not retrieve a group name.\n"};
    }

    hid_t group_id = check_hdf5_status(H5Gopen(root_group_id, group_name.c_str(), H5P_DEFAULT),
                                       std::format("Error: Could not open HDF5 group:{} \n", group_name));

    hid_t dataset_id = check_hdf5_status(H5Dopen(group_id, "Interaction", H5P_DEFAULT),
                                         "Error: Could not open HDF5 interaction dataset \n");

    hid_t dataspace_id = check_hdf5_status(H5Dget_space(dataset_id), "Could not get groups interaction dataspace");

    hsize_t dims[1];
    H5Sget_simple_extent_dims(dataspace_id, dims, nullptr);

    if (dims[0] != len_interaction) {
      // Dimension mismatch, skip this dataset
      H5Sclose(dataspace_id);
      H5Dclose(dataset_id);
      H5Gclose(group_id);
      throw std::runtime_error{
        std::format("Interaction length {} does not coincide with length of interaction written to file {} \n",
                    len_interaction,
                    dims[0])};
    }

    std::vector<signed char> interaction(len_interaction);
    H5Dread(dataset_id, H5T_NATIVE_SCHAR, H5S_ALL, H5S_ALL, H5P_DEFAULT, interaction.data());
    h_interactions_dest[std::stoi(group_name)] = std::move(interaction);

    H5Sclose(dataspace_id);
    H5Dclose(dataset_id);
    H5Gclose(group_id);
  }

  // Close the root group and file
  H5Gclose(root_group_id);
  H5Fclose(file_id);
}

void wl::read_prerun_lattices(std::map<int, std::map<int, std::vector<signed char>>>& h_lattices_dest,
                              const wl::prerun_options options,
                              const std::string_view file_path)
{
  auto len_lattice = static_cast<std::size_t>(options.X * options.Y);
  std::string lattice_path = std::string(file_path) + "/prerun_results.h5";

  hid_t faplist_id = H5Pcreate(H5P_FILE_ACCESS);
  H5Pset_fapl_core(faplist_id, 1, false);
  hid_t file_id = check_hdf5_status(H5Fopen(lattice_path.c_str(), H5F_ACC_RDONLY, faplist_id),
                                    std::format("Error: Could not open HDF5 file:{} \n", lattice_path));

  hid_t root_group_id =
    check_hdf5_status(H5Gopen(file_id, "/Lattice", H5P_DEFAULT), "Error: Could not open root group \n");

  hsize_t num_disorder_samples = 0;
  H5Gget_num_objs(root_group_id, &num_disorder_samples);

  for (hsize_t i = 0; i < num_disorder_samples; ++i) {
    const std::size_t disorder_sample_name_len = 256;
    std::string disorder_sample_name(disorder_sample_name_len, '\0');

    check_hdf5_status(H5Gget_objname_by_idx(root_group_id, i, &disorder_sample_name[0], disorder_sample_name_len),
                      "Error: Could not retrieve a group name.\n");

    hid_t disorder_group_id = check_hdf5_status(H5Gopen2(root_group_id, disorder_sample_name.c_str(), H5P_DEFAULT),
                                                "Failed to open disorder group \n");

    hsize_t num_energy_intervals = 0;
    H5Gget_num_objs(disorder_group_id, &num_energy_intervals);

    for (hsize_t j = 0; j < num_energy_intervals; ++j) {
      const std::size_t energy_name_len = 256;
      std::string energy_name(energy_name_len, '\0');

      check_hdf5_status(H5Gget_objname_by_idx(disorder_group_id, j, &energy_name[0], energy_name_len),
                        "Error: Could not retrieve a group name.\n");

      hid_t energy_group_id = check_hdf5_status(H5Gopen2(disorder_group_id, energy_name.c_str(), H5P_DEFAULT),
                                                "Could not open energy subgroup \n");

      hid_t lattice_dataset_id =
        check_hdf5_status(H5Dopen2(energy_group_id, "Lattice", H5P_DEFAULT), "Could not open lattice dataset \n");

      hid_t lattice_dataspace_id =
        check_hdf5_status(H5Dget_space(lattice_dataset_id), "Could not get lattice dataspace \n");

      hsize_t dims[1];
      H5Sget_simple_extent_dims(lattice_dataspace_id, dims, NULL); // Get the size of the lattice data

      if (dims[0] != len_lattice) {
        throw std::runtime_error{"Error: lattice len from read does not coincide with theoretical lattice len.\n"};
      }

      std::vector<signed char> lattice(len_lattice);

      check_hdf5_status(H5Dread(lattice_dataset_id, H5T_NATIVE_SCHAR, H5S_ALL, H5S_ALL, H5P_DEFAULT, lattice.data()),
                        "Could not read lattice \n");

      int energy_value = std::stoi(energy_name);
      int disorder_sample = std::stoi(disorder_sample_name);

      h_lattices_dest[disorder_sample][energy_value] = lattice;

      H5Dclose(lattice_dataset_id);
      H5Gclose(energy_group_id);
    }
    H5Gclose(disorder_group_id);
  }
  H5Gclose(root_group_id);
  H5Fclose(file_id);
}

void wl::read_prerun_lattices(std::map<int, std::map<int, std::vector<signed char>>>& h_lattices_dest,
                              const wl::mainrun_options& options,
                              const std::string_view file_path)
{
  auto len_lattice = static_cast<std::size_t>(options.X * options.Y);
  std::string lattice_path = std::string(file_path) + "/prerun_results.h5";

  hid_t faplist_id = H5Pcreate(H5P_FILE_ACCESS);
  H5Pset_fapl_core(faplist_id, 1, false);
  hid_t file_id = check_hdf5_status(H5Fopen(lattice_path.c_str(), H5F_ACC_RDONLY, faplist_id),
                                    std::format("Error: Could not open HDF5 file:{} \n", lattice_path));

  hid_t root_group_id =
    check_hdf5_status(H5Gopen(file_id, "/Lattice", H5P_DEFAULT), "Error: Could not open root group \n");

  hsize_t num_disorder_samples = 0;
  H5Gget_num_objs(root_group_id, &num_disorder_samples);

  for (hsize_t i = 0; i < num_disorder_samples; ++i) {
    const std::size_t disorder_sample_name_len = 256;
    std::string disorder_sample_name(disorder_sample_name_len, '\0');

    check_hdf5_status(H5Gget_objname_by_idx(root_group_id, i, &disorder_sample_name[0], disorder_sample_name_len),
                      "Error: Could not retrieve a group name.\n");

    hid_t disorder_group_id = check_hdf5_status(H5Gopen2(root_group_id, disorder_sample_name.c_str(), H5P_DEFAULT),
                                                "Failed to open disorder group \n");

    hsize_t num_energy_intervals = 0;
    H5Gget_num_objs(disorder_group_id, &num_energy_intervals);

    for (hsize_t j = 0; j < num_energy_intervals; ++j) {
      const std::size_t energy_name_len = 256;
      std::string energy_name(energy_name_len, '\0');

      check_hdf5_status(H5Gget_objname_by_idx(disorder_group_id, j, &energy_name[0], energy_name_len),
                        "Error: Could not retrieve a group name.\n");

      hid_t energy_group_id = check_hdf5_status(H5Gopen2(disorder_group_id, energy_name.c_str(), H5P_DEFAULT),
                                                "Could not open energy subgroup \n");

      hid_t lattice_dataset_id =
        check_hdf5_status(H5Dopen2(energy_group_id, "Lattice", H5P_DEFAULT), "Could not open lattice dataset \n");

      hid_t lattice_dataspace_id =
        check_hdf5_status(H5Dget_space(lattice_dataset_id), "Could not get lattice dataspace \n");

      hsize_t dims[1];
      H5Sget_simple_extent_dims(lattice_dataspace_id, dims, NULL); // Get the size of the lattice data

      if (dims[0] != len_lattice) {
        throw std::runtime_error{"Error: lattice len from read does not coincide with theoretical lattice len.\n"};
      }

      std::vector<signed char> lattice(len_lattice);

      check_hdf5_status(H5Dread(lattice_dataset_id, H5T_NATIVE_SCHAR, H5S_ALL, H5S_ALL, H5P_DEFAULT, lattice.data()),
                        "Could not read lattice \n");

      int energy_value = std::stoi(energy_name);
      int disorder_sample = std::stoi(disorder_sample_name);

      h_lattices_dest[disorder_sample][energy_value] = lattice;

      H5Dclose(lattice_dataset_id);
      H5Gclose(energy_group_id);
    }
    H5Gclose(disorder_group_id);
  }
  H5Gclose(root_group_id);
  H5Fclose(file_id);
}

void wl::get_energy_spectrum_information_from_prerun_results(
  std::vector<signed char>& h_expected_energy_spectrum,
  std::vector<int>& h_offset_energy_spectrum,
  std::vector<int>& h_len_energy_spectrum,
  std::vector<int>& E_min,
  std::vector<int>& E_max,
  int total_len_energy_spectrum,
  const std::map<int, std::vector<unsigned char>>& prerun_histograms,
  const wl::mainrun_options& options)
{
  for (int key = 0; key < options.num_interactions; key++) {
    auto it = prerun_histograms.find(key);
    if (it != prerun_histograms.end()) {
      const auto& histogram = it->second;
      auto first_one = std::find(histogram.begin(), histogram.end(), 1);
      auto first_one_index = std::distance(histogram.begin(), first_one);
      if (first_one != histogram.end()) {
        E_min.push_back(-2 * options.X * options.Y + static_cast<int>(first_one_index));
      } else {
        throw std::runtime_error{"Error: Histogram does not contain 1.\n"};
      }
      auto last_one = std::find(histogram.rbegin(), histogram.rend(), 1);
      auto last_one_index =
        histogram.size() - static_cast<std::size_t>(std::distance(histogram.rbegin(), last_one)) - 1;
      if (last_one != histogram.rend()) {
        E_max.push_back(-2 * options.X * options.Y + static_cast<int>(last_one_index));
      } else {
        throw std::runtime_error{"Error: Histogram does not contain 1.\n"};
      }

      h_expected_energy_spectrum.insert(
        h_expected_energy_spectrum.end(), first_one, histogram.begin() + static_cast<int>(last_one_index) + 1);
      h_offset_energy_spectrum.push_back(total_len_energy_spectrum);
      total_len_energy_spectrum += static_cast<int>(last_one_index) - static_cast<int>(first_one_index) + 1;
      h_len_energy_spectrum.push_back(static_cast<int>(last_one_index) - static_cast<int>(first_one_index) + 1);
    } else {
      throw std::runtime_error{std::format("Error: Histogram for interaction {} was not found.\n", key)};
    }
  }
}

void wl::get_interaction_from_prerun_results(std::vector<signed char>& h_interactions,
                                             const std::map<int, std::vector<signed char>>& prerun_interactions,
                                             const wl::mainrun_options& options)
{
  for (int key = 0; key < options.num_interactions; key++) {
    auto iterator = prerun_interactions.find(key);
    if (iterator != prerun_interactions.end()) {
      const auto& interaction = iterator->second;
      int length = static_cast<int>(interaction.size());
      if (length != 2 * options.X * options.Y) {
        throw std::runtime_error{"Error: Interaction not of appropriate length for lattice dim.\n"};
      }
      h_interactions.insert(h_interactions.end(), interaction.begin(), interaction.end());
    } else {
      throw std::runtime_error{std::format("Error: Interaction for disorder offset {} was not found.\n", key)};
    }
  }
}

void wl::get_lattice_from_prerun_results(std::vector<std::int8_t>& h_lattices,
                                         const std::map<int, std::map<int, std::vector<signed char>>>& prerun_lattices,
                                         const std::vector<int>& h_start,
                                         const std::vector<int>& h_end,
                                         const wl::mainrun_options& options)
{
  for (int disorder_key = 0; disorder_key < options.num_interactions; disorder_key++) {
    auto iterator = prerun_lattices.find(disorder_key);
    if (iterator != prerun_lattices.end()) {
      const std::map<int, std::vector<signed char>>& energy_lattice_map = iterator->second;

      // temp storage inits
      std::vector<int> run_start(h_start.begin() + disorder_key * options.num_intervals,
                                 h_start.begin() + disorder_key * options.num_intervals + options.num_intervals);
      std::vector<int> run_end(h_end.begin() + disorder_key * options.num_intervals,
                               h_end.begin() + disorder_key * options.num_intervals + options.num_intervals);

      for (size_t interval_iterator = 0; interval_iterator < static_cast<size_t>(options.num_intervals);
           interval_iterator++) {
        bool found_energy_in_interval = false;
        for (const auto& [energy_key, lattice_vector] : energy_lattice_map) {
          if (energy_key >= run_start[interval_iterator] && energy_key <= run_end[interval_iterator]) {
            found_energy_in_interval = true;
            for (int walker_per_interval_iterator = 0; walker_per_interval_iterator < options.walker_per_interval;
                 walker_per_interval_iterator++) {
              h_lattices.insert(h_lattices.end(), lattice_vector.begin(), lattice_vector.end());
            }
            break;
          }
        }
        if (!found_energy_in_interval) {
          throw std::runtime_error{
            std::format("Error: Did not find lattice configuration with suiting energy for interval [{}, {}] \n",
                        run_start[interval_iterator],
                        run_end[interval_iterator])};
        }
      }
    } else {
      throw std::runtime_error{std::format("Error: No Lattices found for disorder offset {}.\n", disorder_key)};
    }
  }
}
