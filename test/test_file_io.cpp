#define BOOST_TEST_MODULE wl_tests

#include <boost/test/unit_test.hpp>

#include <hdf5.h>

#include <cstddef>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

#include "../src/file_io.hpp"
#include "../src/program_options.hpp"
#include "../src/utils.hpp"

// annonymous namespace seems to be practice for isolating the fixture from the global namespace
namespace {
const std::string test_file_path = "./tmp/test_files";

hid_t create_hdf5_file(std::string_view file_path)
{
  std::string histogram_path = std::string(file_path) + "/prerun_results.h5";

  return H5Fcreate(histogram_path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
}

struct TestFixture
{
  // Cleanup teardown fixture that runs after each test to remove the test files
  ~TestFixture()
  {
    if (std::filesystem::exists(test_file_path)) {
      std::filesystem::remove_all(test_file_path);
    }
  }
};

// Register the fixture globally
BOOST_TEST_GLOBAL_FIXTURE(TestFixture);
}

BOOST_AUTO_TEST_SUITE(file_io)

BOOST_AUTO_TEST_CASE(histogram_write_read_test)
{
  const std::string path = test_file_path + "/hist";
  wl::create_directories(path);

  BOOST_REQUIRE(std::filesystem::exists(path));

  const std::vector<unsigned long long> write_histogram = {99, 0, 0, 0, 98, 0, 0, 97, 0, 96};

  std::map<int, std::vector<unsigned char>> expected_histogram;
  expected_histogram[0] = {1, 0, 0, 0, 1};
  expected_histogram[1] = {0, 0, 1, 0, 1};

  wl::prerun_options options;
  options.num_interactions = 2;

  const std::size_t len_histogram = 5;

  hid_t file_id = create_hdf5_file(path);
  BOOST_REQUIRE_NO_THROW(wl::write_prerun_histogram_results(write_histogram, options, len_histogram, file_id));
  H5Fclose(file_id);

  std::map<int, std::vector<unsigned char>> read_histogram_dest;
  BOOST_REQUIRE_NO_THROW(wl::read_prerun_histograms(read_histogram_dest, path, len_histogram));

  BOOST_REQUIRE_EQUAL(expected_histogram.size(), read_histogram_dest.size());

  BOOST_CHECK_EQUAL_COLLECTIONS(expected_histogram[0].begin(),
                                expected_histogram[0].end(),
                                read_histogram_dest[0].begin(),
                                read_histogram_dest[0].end());

  BOOST_CHECK_EQUAL_COLLECTIONS(expected_histogram[1].begin(),
                                expected_histogram[1].end(),
                                read_histogram_dest[1].begin(),
                                read_histogram_dest[1].end());
}

BOOST_AUTO_TEST_CASE(interaction_write_read_test)
{
  const std::string path = test_file_path + "/int";
  wl::create_directories(path);

  BOOST_REQUIRE(std::filesystem::exists(path));

  std::vector<signed char> write_interaction = {
    1, -1, 1, 1, 1, 1, -1, 1, 1, 1, -1, 1, 1, -1, -1, 1,  1,  1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, -1, 1,
    1, 1,  1, 1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1,  1,  -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1,  -1};

  std::map<int, std::vector<signed char>> expected_interaction;
  expected_interaction[0] = {1, -1, 1, 1, 1, 1, -1, 1, 1, 1,  -1, 1, 1, -1, -1, 1,
                             1, 1,  1, 1, 1, 1, 1,  1, 1, -1, 1,  1, 1, 1,  -1, 1};

  expected_interaction[1] = {1,  1, 1, 1, 1, 1, 1, 1, 1, 1,  1, 1, 1, 1, 1, -1,
                             -1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, -1};

  wl::prerun_options options;
  options.X = 4;
  options.Y = 4;
  options.num_interactions = 2;

  hid_t file_id = create_hdf5_file(path);
  BOOST_REQUIRE_NO_THROW(wl::write_prerun_interaction_results(write_interaction, options, file_id));
  H5Fclose(file_id);

  std::map<int, std::vector<signed char>> read_interaction_dest;
  BOOST_REQUIRE_NO_THROW(wl::read_prerun_interactions(read_interaction_dest, path, options));

  BOOST_REQUIRE_EQUAL(read_interaction_dest.size(), expected_interaction.size());

  BOOST_CHECK_EQUAL_COLLECTIONS(expected_interaction[0].begin(),
                                expected_interaction[0].end(),
                                read_interaction_dest[0].begin(),
                                read_interaction_dest[0].end());

  BOOST_CHECK_EQUAL_COLLECTIONS(expected_interaction[1].begin(),
                                expected_interaction[1].end(),
                                read_interaction_dest[1].begin(),
                                read_interaction_dest[1].end());
}

BOOST_AUTO_TEST_CASE(lattice_write_read_test)
{
  wl::prerun_options options;
  options.X = 4;
  options.Y = 4;
  options.num_interactions = 2;
  options.num_intervals = 4;

  const std::string path = test_file_path + "/lat";
  wl::create_directories(path);

  BOOST_REQUIRE(std::filesystem::exists(path));

  std::vector<signed char> write_lattice = {1, -1, 1, 1,  1,  1,  -1, 1,  1,
                                            1, -1, 1, 1,  -1, -1, 1,

                                            1, 1,  1, -1, 1,  1,  1,  -1, -1,
                                            1, 1,  1, 1,  1,  1,  1,

                                            1, 1,  1, -1, -1, 1,  1,  1,  1,
                                            1, -1, 1, 1,  -1, 1,  1,

                                            0, 0,  0, 0,  0,  0,  0,  0,  0,
                                            0, 0,  0, 0,  0,  0,  0,

                                            1, -1, 1, 1,  1,  1,  -1, 1,  1,
                                            1, -1, 1, 1,  -1, -1, 1,

                                            1, 1,  1, -1, 1,  1,  1,  -1, -1,
                                            1, 1,  1, 1,  1,  1,  1,

                                            0, 0,  0, 0,  0,  0,  0,  0,  0,
                                            0, 0,  0, 0,  0,  0,  0,

                                            1, 1,  1, -1, -1, 1,  1,  1,  1,
                                            1, -1, 1, 1,  -1, 1,  1

  };

  const std::vector<int> h_energies = {0, 10, 20, 30, 40, 50, 60, 70};

  std::map<int, std::map<int, std::vector<signed char>>> expected_lattices;

  expected_lattices[0][h_energies[0]] = {1, -1, 1, 1, 1, 1, -1, 1, 1, 1, -1, 1, 1, -1, -1, 1};

  expected_lattices[0][h_energies[1]] = {1, 1, 1, -1, 1, 1, 1, -1, -1, 1, 1, 1, 1, 1, 1, 1};

  expected_lattices[0][h_energies[2]] = {1, 1, 1, -1, -1, 1, 1, 1, 1, 1, -1, 1, 1, -1, 1, 1};

  expected_lattices[1][h_energies[4]] = {1, -1, 1, 1, 1, 1, -1, 1, 1, 1, -1, 1, 1, -1, -1, 1};

  expected_lattices[1][h_energies[5]] = {1, 1, 1, -1, 1, 1, 1, -1, -1, 1, 1, 1, 1, 1, 1, 1};

  expected_lattices[1][h_energies[7]] = {1, 1, 1, -1, -1, 1, 1, 1, 1, 1, -1, 1, 1, -1, 1, 1};

  hid_t file_id = create_hdf5_file(path);
  BOOST_REQUIRE_NO_THROW(wl::write_prerun_lattice_results(write_lattice, h_energies, options, file_id));
  H5Fclose(file_id);

  std::map<int, std::map<int, std::vector<signed char>>> read_lattice_dest;
  BOOST_REQUIRE_NO_THROW(wl::read_prerun_lattices(read_lattice_dest, options, path));

  BOOST_REQUIRE_EQUAL(read_lattice_dest.size(), expected_lattices.size());

  BOOST_CHECK_EQUAL_COLLECTIONS(expected_lattices[0][0].begin(),
                                expected_lattices[0][0].end(),
                                read_lattice_dest[0][0].begin(),
                                read_lattice_dest[0][0].end());

  BOOST_CHECK_EQUAL_COLLECTIONS(expected_lattices[0][10].begin(),
                                expected_lattices[0][10].end(),
                                read_lattice_dest[0][10].begin(),
                                read_lattice_dest[0][10].end());

  BOOST_CHECK_EQUAL_COLLECTIONS(expected_lattices[0][20].begin(),
                                expected_lattices[0][20].end(),
                                read_lattice_dest[0][20].begin(),
                                read_lattice_dest[0][20].end());

  BOOST_CHECK_EQUAL_COLLECTIONS(expected_lattices[1][40].begin(),
                                expected_lattices[1][40].end(),
                                read_lattice_dest[1][40].begin(),
                                read_lattice_dest[1][40].end());

  BOOST_CHECK_EQUAL_COLLECTIONS(expected_lattices[1][50].begin(),
                                expected_lattices[1][50].end(),
                                read_lattice_dest[1][50].begin(),
                                read_lattice_dest[1][50].end());

  BOOST_CHECK_EQUAL_COLLECTIONS(expected_lattices[1][70].begin(),
                                expected_lattices[1][70].end(),
                                read_lattice_dest[1][70].begin(),
                                read_lattice_dest[1][70].end());

  BOOST_CHECK(read_lattice_dest[1][60].empty());
  BOOST_CHECK(read_lattice_dest[0][30].empty());
}

BOOST_AUTO_TEST_CASE(prerun_write_read_test)
{
  // Init directory
  wl::create_directories(test_file_path);
  BOOST_REQUIRE(std::filesystem::exists(test_file_path));

  // Inputs
  const std::vector<unsigned long long> write_histogram = {99, 0, 0, 0, 98, 0, 0, 97, 0, 96};
  std::vector<signed char> write_interaction = {
    1, -1, 1, 1, 1, 1, -1, 1, 1, 1, -1, 1, 1, -1, -1, 1,  1,  1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, -1, 1,
    1, 1,  1, 1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1,  1,  -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1,  -1};
  std::vector<signed char> write_lattice = {1, -1, 1, 1,  1,  1,  -1, 1,  1,
                                            1, -1, 1, 1,  -1, -1, 1,

                                            1, 1,  1, -1, 1,  1,  1,  -1, -1,
                                            1, 1,  1, 1,  1,  1,  1,

                                            1, 1,  1, -1, -1, 1,  1,  1,  1,
                                            1, -1, 1, 1,  -1, 1,  1,

                                            0, 0,  0, 0,  0,  0,  0,  0,  0,
                                            0, 0,  0, 0,  0,  0,  0,

                                            1, -1, 1, 1,  1,  1,  -1, 1,  1,
                                            1, -1, 1, 1,  -1, -1, 1,

                                            1, 1,  1, -1, 1,  1,  1,  -1, -1,
                                            1, 1,  1, 1,  1,  1,  1,

                                            0, 0,  0, 0,  0,  0,  0,  0,  0,
                                            0, 0,  0, 0,  0,  0,  0,

                                            1, 1,  1, -1, -1, 1,  1,  1,  1,
                                            1, -1, 1, 1,  -1, 1,  1

  };
  const std::vector<int> h_energies = {0, 10, 20, 30, 40, 50, 60, 70};

  // Expectations
  std::map<int, std::vector<unsigned char>> expected_histogram;
  expected_histogram[0] = {1, 0, 0, 0, 1};
  expected_histogram[1] = {0, 0, 1, 0, 1};
  std::map<int, std::vector<signed char>> expected_interaction;
  expected_interaction[0] = {1, -1, 1, 1, 1, 1, -1, 1, 1, 1,  -1, 1, 1, -1, -1, 1,
                             1, 1,  1, 1, 1, 1, 1,  1, 1, -1, 1,  1, 1, 1,  -1, 1};

  expected_interaction[1] = {1,  1, 1, 1, 1, 1, 1, 1, 1, 1,  1, 1, 1, 1, 1, -1,
                             -1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, -1};
  std::map<int, std::map<int, std::vector<signed char>>> expected_lattices;

  expected_lattices[0][h_energies[0]] = {1, -1, 1, 1, 1, 1, -1, 1, 1, 1, -1, 1, 1, -1, -1, 1};

  expected_lattices[0][h_energies[1]] = {1, 1, 1, -1, 1, 1, 1, -1, -1, 1, 1, 1, 1, 1, 1, 1};

  expected_lattices[0][h_energies[2]] = {1, 1, 1, -1, -1, 1, 1, 1, 1, 1, -1, 1, 1, -1, 1, 1};

  expected_lattices[1][h_energies[4]] = {1, -1, 1, 1, 1, 1, -1, 1, 1, 1, -1, 1, 1, -1, -1, 1};

  expected_lattices[1][h_energies[5]] = {1, 1, 1, -1, 1, 1, 1, -1, -1, 1, 1, 1, 1, 1, 1, 1};

  expected_lattices[1][h_energies[7]] = {1, 1, 1, -1, -1, 1, 1, 1, 1, 1, -1, 1, 1, -1, 1, 1};

  // Parameter
  wl::prerun_options options;
  options.X = 4;
  options.Y = 4;
  options.num_interactions = 2;
  options.num_intervals = 4;
  const std::size_t len_histogram = 5;

  // Write
  BOOST_REQUIRE_NO_THROW(wl::write_prerun_results(
    write_histogram, write_interaction, write_lattice, h_energies, options, len_histogram, test_file_path));

  // Destinations
  std::map<int, std::vector<unsigned char>> read_histogram_dest;
  std::map<int, std::vector<signed char>> read_interaction_dest;
  std::map<int, std::map<int, std::vector<signed char>>> read_lattice_dest;

  // Read
  BOOST_REQUIRE_NO_THROW(wl::read_prerun_histograms(read_histogram_dest, test_file_path, len_histogram));
  BOOST_REQUIRE_NO_THROW(wl::read_prerun_interactions(read_interaction_dest, test_file_path, options));
  BOOST_REQUIRE_NO_THROW(wl::read_prerun_lattices(read_lattice_dest, options, test_file_path));

  // Histogram checks
  BOOST_REQUIRE_EQUAL(expected_histogram.size(), read_histogram_dest.size());

  BOOST_CHECK_EQUAL_COLLECTIONS(expected_histogram[0].begin(),
                                expected_histogram[0].end(),
                                read_histogram_dest[0].begin(),
                                read_histogram_dest[0].end());

  BOOST_CHECK_EQUAL_COLLECTIONS(expected_histogram[1].begin(),
                                expected_histogram[1].end(),
                                read_histogram_dest[1].begin(),
                                read_histogram_dest[1].end());
  // Interaction checks
  BOOST_REQUIRE_EQUAL(read_interaction_dest.size(), expected_interaction.size());

  BOOST_CHECK_EQUAL_COLLECTIONS(expected_interaction[0].begin(),
                                expected_interaction[0].end(),
                                read_interaction_dest[0].begin(),
                                read_interaction_dest[0].end());

  BOOST_CHECK_EQUAL_COLLECTIONS(expected_interaction[1].begin(),
                                expected_interaction[1].end(),
                                read_interaction_dest[1].begin(),
                                read_interaction_dest[1].end());

  // Lattice cheks
  BOOST_REQUIRE_EQUAL(read_lattice_dest.size(), expected_lattices.size());

  BOOST_CHECK_EQUAL_COLLECTIONS(expected_lattices[0][0].begin(),
                                expected_lattices[0][0].end(),
                                read_lattice_dest[0][0].begin(),
                                read_lattice_dest[0][0].end());

  BOOST_CHECK_EQUAL_COLLECTIONS(expected_lattices[0][10].begin(),
                                expected_lattices[0][10].end(),
                                read_lattice_dest[0][10].begin(),
                                read_lattice_dest[0][10].end());

  BOOST_CHECK_EQUAL_COLLECTIONS(expected_lattices[0][20].begin(),
                                expected_lattices[0][20].end(),
                                read_lattice_dest[0][20].begin(),
                                read_lattice_dest[0][20].end());

  BOOST_CHECK_EQUAL_COLLECTIONS(expected_lattices[1][40].begin(),
                                expected_lattices[1][40].end(),
                                read_lattice_dest[1][40].begin(),
                                read_lattice_dest[1][40].end());

  BOOST_CHECK_EQUAL_COLLECTIONS(expected_lattices[1][50].begin(),
                                expected_lattices[1][50].end(),
                                read_lattice_dest[1][50].begin(),
                                read_lattice_dest[1][50].end());

  BOOST_CHECK_EQUAL_COLLECTIONS(expected_lattices[1][70].begin(),
                                expected_lattices[1][70].end(),
                                read_lattice_dest[1][70].begin(),
                                read_lattice_dest[1][70].end());

  BOOST_CHECK(read_lattice_dest[1][60].empty());
  BOOST_CHECK(read_lattice_dest[0][30].empty());
}

BOOST_AUTO_TEST_CASE(GetEnergySpectrumInformationFromPrerunResults_Success)
{
  const int num_interactions = 2;
  const int X_dim = 5;
  const int Y_dim = 5;

  std::vector<signed char> h_expected_energy_spectrum;
  std::vector<int> h_offset_energy_spectrum;
  std::vector<int> h_len_energy_spectrum;
  std::vector<int> E_min;
  std::vector<int> E_max;
  int total_len_energy_spectrum = 0;

  std::map<int, std::vector<unsigned char>> prerun_histograms = {{0, {0, 1, 1, 0}}, {1, {0, 0, 1, 1}}};

  wl::mainrun_options wl_options;
  wl_options.num_interactions = num_interactions;
  wl_options.X = X_dim;
  wl_options.Y = Y_dim;

  BOOST_REQUIRE_NO_THROW(wl::get_energy_spectrum_information_from_prerun_results(h_expected_energy_spectrum,
                                                                                 h_offset_energy_spectrum,
                                                                                 h_len_energy_spectrum,
                                                                                 E_min,
                                                                                 E_max,
                                                                                 total_len_energy_spectrum,
                                                                                 prerun_histograms,
                                                                                 wl_options));

  // Expectations
  std::vector<signed char> expected_h_expected_energy_spectrum = {1, 1, 1, 1};
  std::vector<int> expected_E_min = {-49, -48};                // Minimum energy values per interaction
  std::vector<int> expected_E_max = {-48, -47};                // Maximum energy values per interaction
  std::vector<int> expected_h_offset_energy_spectrum = {0, 2}; // Offset per interaction
  std::vector<int> expected_h_len_energy_spectrum = {2, 2};

  BOOST_CHECK_EQUAL_COLLECTIONS(h_expected_energy_spectrum.begin(),
                                h_expected_energy_spectrum.end(),
                                expected_h_expected_energy_spectrum.begin(),
                                expected_h_expected_energy_spectrum.end());

  BOOST_CHECK_EQUAL_COLLECTIONS(E_min.begin(), E_min.end(), expected_E_min.begin(), expected_E_min.end());

  BOOST_CHECK_EQUAL_COLLECTIONS(E_max.begin(), E_max.end(), expected_E_max.begin(), expected_E_max.end());

  BOOST_CHECK_EQUAL_COLLECTIONS(h_offset_energy_spectrum.begin(),
                                h_offset_energy_spectrum.end(),
                                expected_h_offset_energy_spectrum.begin(),
                                expected_h_offset_energy_spectrum.end());

  BOOST_CHECK_EQUAL_COLLECTIONS(h_len_energy_spectrum.begin(),
                                h_len_energy_spectrum.end(),
                                expected_h_len_energy_spectrum.begin(),
                                expected_h_len_energy_spectrum.end());
}

BOOST_AUTO_TEST_CASE(GetEnergySpectrumInformationFromPrerunResults_MissingHistogram)
{
  const int num_interactions = 2;
  const int X_dim = 5;
  const int Y_dim = 5;

  std::vector<signed char> h_expected_energy_spectrum;
  std::vector<int> h_offset_energy_spectrum;
  std::vector<int> h_len_energy_spectrum;
  std::vector<int> E_min;
  std::vector<int> E_max;
  int total_len_energy_spectrum = 0;

  std::map<int, std::vector<unsigned char>> prerun_histograms = {{0, {0, 1, 1, 0}}};

  wl::mainrun_options wl_options;
  wl_options.num_interactions = num_interactions;
  wl_options.X = X_dim;
  wl_options.Y = Y_dim;

  BOOST_CHECK_THROW(wl::get_energy_spectrum_information_from_prerun_results(h_expected_energy_spectrum,
                                                                            h_offset_energy_spectrum,
                                                                            h_len_energy_spectrum,
                                                                            E_min,
                                                                            E_max,
                                                                            total_len_energy_spectrum,
                                                                            prerun_histograms,
                                                                            wl_options),
                    std::runtime_error);
}

BOOST_AUTO_TEST_CASE(GetInteractionFromPrerunResults_Success)
{
  const int num_interactions = 2;
  const int X_dim = 2;
  const int Y_dim = 2;

  std::vector<signed char> h_interactions;
  std::map<int, std::vector<signed char>> prerun_interactions = {{0, {1, -1, 1, -1, 1, -1, 1, -1}},
                                                                 {1, {1, 1, 1, -1, 1, -1, 1, -1}}};

  wl::mainrun_options wl_options;
  wl_options.num_interactions = num_interactions;
  wl_options.X = X_dim;
  wl_options.Y = Y_dim;

  BOOST_REQUIRE_NO_THROW(wl::get_interaction_from_prerun_results(h_interactions, prerun_interactions, wl_options));

  std::vector<signed char> expected_h_interactions = {1, -1, 1, -1, 1, -1, 1, -1, 1, 1, 1, -1, 1, -1, 1, -1};

  BOOST_CHECK_EQUAL_COLLECTIONS(
    h_interactions.begin(), h_interactions.end(), expected_h_interactions.begin(), expected_h_interactions.end());
}

BOOST_AUTO_TEST_CASE(GetInteractionFromPrerunResults_MissingInteraction)
{
  const int num_interactions = 2;
  const int X_dim = 2;
  const int Y_dim = 2;

  std::vector<signed char> h_interactions;
  std::map<int, std::vector<signed char>> prerun_interactions = {{0, {1, -1, 1, 1, 1, -1, 1, 1}}};

  wl::mainrun_options wl_options;
  wl_options.num_interactions = num_interactions;
  wl_options.X = X_dim;
  wl_options.Y = Y_dim;

  BOOST_CHECK_THROW(wl::get_interaction_from_prerun_results(h_interactions, prerun_interactions, wl_options),
                    std::runtime_error);
}

BOOST_AUTO_TEST_CASE(GetInteractionFromPrerunResults_WrongInteractionLengthForLatticeDim)
{
  const int num_interactions = 2;
  const int X_dim = 5;
  const int Y_dim = 5;

  std::vector<signed char> h_interactions;
  std::map<int, std::vector<signed char>> prerun_interactions = {{0, {1, -1, 1}}, {1, {1, -1, 1}}};

  wl::mainrun_options wl_options;
  wl_options.num_interactions = num_interactions;
  wl_options.X = X_dim;
  wl_options.Y = Y_dim;

  BOOST_CHECK_THROW(wl::get_interaction_from_prerun_results(h_interactions, prerun_interactions, wl_options),
                    std::runtime_error);
}

BOOST_AUTO_TEST_CASE(GetLatticeFromPrerunResults_Success)
{
  const int num_interactions = 2;
  const int num_intervals = 2;
  const int walker_per_interval = 2;
  const int X_dim = 5;
  const int Y_dim = 5;

  std::vector<std::int8_t> h_lattices;
  wl::mainrun_options wl_options;
  wl_options.num_interactions = num_interactions;
  wl_options.num_intervals = num_intervals;
  wl_options.walker_per_interval = walker_per_interval;
  wl_options.X = X_dim;
  wl_options.Y = Y_dim;

  // Input
  std::map<int, std::map<int, std::vector<signed char>>> prerun_lattices{
    {0, {{-2, {1, -1, 1, 1}}, {2, {1, 1, 1, -1}}}}, {1, {{-4, {-1, 1, 1, 1}}, {4, {1, 1, -1, 1}}}}};

  std::vector<int> h_start{-2, 0, -4, 0};
  std::vector<int> h_end{-1, 2, -1, 4};

  // Expected
  std::vector<std::int8_t> expected_lattices = {1,  -1, 1, 1, 1,  -1, 1, 1, 1, 1, 1,  -1, 1, 1, 1,  -1,
                                                -1, 1,  1, 1, -1, 1,  1, 1, 1, 1, -1, 1,  1, 1, -1, 1};

  BOOST_REQUIRE_NO_THROW(wl::get_lattice_from_prerun_results(h_lattices, prerun_lattices, h_start, h_end, wl_options));

  // Checks
  BOOST_CHECK_EQUAL_COLLECTIONS(
    h_lattices.begin(), h_lattices.end(), expected_lattices.begin(), expected_lattices.end());
}

BOOST_AUTO_TEST_CASE(GetLatticeFromPrerunResults_MissingLatticeForInterval)
{
  const int num_interactions = 2;
  const int num_intervals = 2;
  const int walker_per_interval = 2;
  const int X_dim = 5;
  const int Y_dim = 5;

  std::vector<std::int8_t> h_lattices;
  wl::mainrun_options wl_options;
  wl_options.num_interactions = num_interactions;
  wl_options.num_intervals = num_intervals;
  wl_options.walker_per_interval = walker_per_interval;
  wl_options.X = X_dim;
  wl_options.Y = Y_dim;

  // Input
  std::map<int, std::map<int, std::vector<signed char>>> prerun_lattices{
    {0, {{-2, {1, -1, 1, 1}}, {-1, {1, 1, 1, -1}}}}, {1, {{-4, {-1, 1, 1, 1}}, {4, {1, 1, -1, 1}}}}};

  std::vector<int> h_start{-2, 0, -4, 0};
  std::vector<int> h_end{-1, 2, -1, 4};

  // Expected
  std::vector<std::int8_t> expected_lattices = {1,  -1, 1, 1, 1,  -1, 1, 1, 1, 1, 1,  -1, 1, 1, 1,  -1,
                                                -1, 1,  1, 1, -1, 1,  1, 1, 1, 1, -1, 1,  1, 1, -1, 1};

  // Checks
  BOOST_CHECK_THROW(wl::get_lattice_from_prerun_results(h_lattices, prerun_lattices, h_start, h_end, wl_options),
                    std::runtime_error);
}

BOOST_AUTO_TEST_SUITE_END()
