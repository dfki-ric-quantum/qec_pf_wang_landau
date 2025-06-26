#define BOOST_TEST_MODULE wl_tests

#include <H5Cpp.h>
#include <boost/test/unit_test.hpp>
#include <filesystem>
#include <hdf5.h>
#include <map>
#include <string>
#include <vector>

#include "../src/utils.hpp"

// annonymous namespace seems to be practice for isolating the fixture from the global namespace
namespace {
const std::string test_file_path = "./tmp/test_files";

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

struct ResultEntry
{
  int key;
  double value;
};

BOOST_AUTO_TEST_SUITE(result_handling)

BOOST_AUTO_TEST_CASE(test_write_results)
{
  // Initialize inputs
  std::vector<std::map<int, double>> result_data = {{{1, 0.5}, {2, 1.5}}, {{3, 2.5}, {4, 3.5}}};

  wl::mainrun_options options;
  options.seed = 42;

  int disorder_id = 100;
  std::string timestamp_group_name = "timestamp_01:01:2024";
  std::string git_info = "release_0001-10-h76if76a-dirty";

  wl::create_directories(test_file_path);

  // Call the function to write the results
  BOOST_REQUIRE_NO_THROW(
    wl::write_results(result_data, options, disorder_id, test_file_path, timestamp_group_name, git_info));

  try {
    H5::H5File file(test_file_path + "/mainrun_results.h5", H5F_ACC_RDONLY);

    std::string disorder_group_name = "/" + std::to_string(disorder_id);
    std::string run_group_name = std::to_string(options.seed);
    std::string dataset_path = disorder_group_name + "/" + run_group_name + "/" + timestamp_group_name + "/log_g";

    H5::DataSet dataset = file.openDataSet(dataset_path);

    // Verify the dataset type and size
    H5::CompType compound_type(sizeof(ResultEntry));
    compound_type.insertMember("Key", HOFFSET(ResultEntry, key), H5::PredType::NATIVE_INT);
    compound_type.insertMember("Value", HOFFSET(ResultEntry, value), H5::PredType::NATIVE_DOUBLE);

    H5::DataSpace dataspace = dataset.getSpace();
    hsize_t dims[1];
    dataspace.getSimpleExtentDims(dims, nullptr);
    BOOST_CHECK_EQUAL(dims[0], result_data[0].size() + result_data[1].size());

    std::vector<ResultEntry> read_data(dims[0]);
    dataset.read(read_data.data(), compound_type);

    // flatten input data for comparison to output from read
    std::vector<ResultEntry> expected_data;
    for (const auto& map : result_data) {
      for (const auto& [key, value] : map) {
        expected_data.push_back({key, value});
      }
    }

    BOOST_CHECK_EQUAL(read_data.size(), expected_data.size());
    for (size_t i = 0; i < expected_data.size(); ++i) {
      BOOST_CHECK_EQUAL(read_data[i].key, expected_data[i].key);
      BOOST_CHECK_CLOSE(read_data[i].value, expected_data[i].value, 1e-6);
    }

    // Validate the attribute written to dataset
    H5::Attribute attr = dataset.openAttribute("CodeVersion");
    H5::StrType str_type(H5::PredType::C_S1, 256);
    std::string attr_value(256, '\0');
    attr.read(str_type, attr_value);
    size_t null_pos = attr_value.find('\0');
    if (null_pos != std::string::npos) {
      attr_value.erase(null_pos);
    }
    // Verify the attribute value
    BOOST_CHECK_EQUAL(attr_value, git_info);

  } catch (std::exception& e) {
    BOOST_FAIL("Standard exception: " + std::string(e.what()));
  } catch (...) {
    BOOST_FAIL("Unknown exception encountered.");
  }
}
BOOST_AUTO_TEST_SUITE_END()