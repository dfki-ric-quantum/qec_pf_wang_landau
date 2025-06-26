#include <boost/test/tools/old/interface.hpp>
#define BOOST_TEST_MODULE wl_tests

#include <algorithm>
#include <cstdint>
#include <vector>

#include <boost/test/unit_test.hpp>
#include <cuda_runtime.h>

#include "../src/cuda_utils.cuh"
#include "../src/wl.cuh"

BOOST_AUTO_TEST_SUITE(init_interaction)

/**
 * Test ratio of flipped interactions compared to error probability with allowed 1% deviation from expected value.
 */
BOOST_AUTO_TEST_CASE(test_ratio_of_flipped_interactions)
{
  const int lattice_dim_x = 10;
  const int lattice_dim_y = 10;
  const std::size_t lattice_size = lattice_dim_x * lattice_dim_y;
  const int num_interactions = 1000;
  const int seed = 1;
  const float prob = 0.15F;

  const int THREADS = 128;
  const int count_all_interactions = num_interactions * lattice_dim_x * lattice_dim_y * 2;
  const int BLOCKS_INIT = (count_all_interactions + THREADS - 1) / THREADS;

  wl::device_ptr<std::int8_t> d_interactions(count_all_interactions);
  wl::init_interactions<<<BLOCKS_INIT, THREADS>>>(d_interactions.data(), lattice_size, num_interactions, seed, prob);

  std::vector<std::int8_t> h_interactions(count_all_interactions, 1);
  d_interactions.device_to_host(h_interactions);

  int count_neg_one_interactions = static_cast<int>(std::count(h_interactions.begin(), h_interactions.end(), -1));
  float neg_one_interactions_x_all_interactions =
    (count_neg_one_interactions != 0) ? static_cast<float>(count_neg_one_interactions) / count_all_interactions : 0;

  BOOST_CHECK_CLOSE(neg_one_interactions_x_all_interactions, prob, 1.0);
}

/**
 * Test expected interaction configuration when horizontal error chain is applied.
 */
BOOST_AUTO_TEST_CASE(test_horizontal_error_cycle)
{
  const int lattice_dim_x = 4;
  const int lattice_dim_y = 4;
  const std::size_t lattice_size = lattice_dim_x * lattice_dim_y;
  const int num_interactions = 2;
  const int seed = 1;
  const float prob = 0.0F;

  const int THREADS = 128;
  const int count_all_interactions = num_interactions * lattice_dim_x * lattice_dim_y * 2;
  const int BLOCKS_INIT = (count_all_interactions + THREADS - 1) / THREADS;

  // clang-format off
    std::vector<std::int8_t> expected_interactions = {1,1,1,1,
                                                      1,1,1,1,
                                                      1,1,1,1,
                                                      1,1,1,1,
                                                     -1,-1,-1,-1,
                                                      1,1,1,1,
                                                      1,1,1,1,
                                                      1,1,1,1,
                                                      1,1,1,1,
                                                      1,1,1,1,
                                                      1,1,1,1,
                                                      1,1,1,1,
                                                      -1,-1,-1,-1,
                                                      1,1,1,1,
                                                      1,1,1,1,
                                                      1,1,1,1};
  // clang-format on

  wl::device_ptr<std::int8_t> d_interactions(count_all_interactions, 0);
  wl::init_interactions<<<BLOCKS_INIT, THREADS>>>(d_interactions.data(), lattice_size, num_interactions, seed, prob);
  wl::apply_x_horizontal_error<<<BLOCKS_INIT, THREADS>>>(
    d_interactions.data(), lattice_dim_x, lattice_dim_y, num_interactions);

  std::vector<std::int8_t> h_interactions(count_all_interactions, 1);
  d_interactions.device_to_host(h_interactions);

  BOOST_CHECK_EQUAL_COLLECTIONS(
    h_interactions.begin(), h_interactions.end(), expected_interactions.begin(), expected_interactions.end());
}

/**
 * Test expected interaction configuration when vertical error chain is applied.
 */
BOOST_AUTO_TEST_CASE(test_vertical_error_cycle)
{
  const int lattice_dim_x = 4;
  const int lattice_dim_y = 4;
  const std::size_t lattice_size = lattice_dim_x * lattice_dim_y;
  const int num_interactions = 2;
  const int seed = 1;
  const float prob = 0;

  const int THREADS = 128;
  const int count_all_interactions = num_interactions * lattice_dim_x * lattice_dim_y * 2;
  const int BLOCKS_INIT = (count_all_interactions + THREADS - 1) / THREADS;

  // clang-format off
    std::vector<std::int8_t> expected_interactions = {-1,1,1,1,
                                                      -1,1,1,1,
                                                      -1,1,1,1,
                                                      -1,1,1,1,
                                                       1,1,1,1,
                                                       1,1,1,1,
                                                       1,1,1,1,
                                                       1,1,1,1,
                                                      -1,1,1,1,
                                                      -1,1,1,1,
                                                      -1,1,1,1,
                                                      -1,1,1,1,
                                                       1,1,1,1,
                                                       1,1,1,1,
                                                       1,1,1,1,
                                                       1,1,1,1};
  // clang-format on

  wl::device_ptr<std::int8_t> d_interactions(count_all_interactions);
  wl::init_interactions<<<BLOCKS_INIT, THREADS>>>(d_interactions.data(), lattice_size, num_interactions, seed, prob);
  wl::apply_x_vertical_error<<<BLOCKS_INIT, THREADS>>>(
    d_interactions.data(), lattice_dim_x, lattice_dim_y, num_interactions);

  std::vector<std::int8_t> h_interactions(count_all_interactions, 1);
  d_interactions.device_to_host(h_interactions);

  BOOST_CHECK_EQUAL_COLLECTIONS(
    h_interactions.begin(), h_interactions.end(), expected_interactions.begin(), expected_interactions.end());
}

/**
 * Test expected interaction configuration when horizontal and vertical error chains are applied.
 */
BOOST_AUTO_TEST_CASE(test_vertical_and_horizontal_error_cycle)
{
  const int lattice_dim_x = 4;
  const int lattice_dim_y = 4;
  const std::size_t lattice_size = lattice_dim_x * lattice_dim_y;
  const int num_interactions = 2;
  const int seed = 1;
  const float prob = 0;

  const int THREADS = 128;
  const int count_all_interactions = num_interactions * lattice_dim_x * lattice_dim_y * 2;
  const int BLOCKS_INIT = (count_all_interactions + THREADS - 1) / THREADS;

  // clang-format off
    std::vector<std::int8_t> expected_interactions = {-1,1,1,1,
                                                      -1,1,1,1,
                                                      -1,1,1,1,
                                                      -1,1,1,1,
                                                      -1,-1,-1,-1,
                                                       1,1,1,1,
                                                       1,1,1,1,
                                                       1,1,1,1,
                                                      -1,1,1,1,
                                                      -1,1,1,1,
                                                      -1,1,1,1,
                                                      -1,1,1,1,
                                                      -1,-1,-1,-1,
                                                       1,1,1,1,
                                                       1,1,1,1,
                                                       1,1,1,1};
  // clang-format on

  wl::device_ptr<std::int8_t> d_interactions(count_all_interactions);

  wl::init_interactions<<<BLOCKS_INIT, THREADS>>>(d_interactions.data(), lattice_size, num_interactions, seed, prob);

  wl::apply_x_vertical_error<<<BLOCKS_INIT, THREADS>>>(
    d_interactions.data(), lattice_dim_x, lattice_dim_y, num_interactions);
  wl::apply_x_horizontal_error<<<BLOCKS_INIT, THREADS>>>(
    d_interactions.data(), lattice_dim_x, lattice_dim_y, num_interactions);

  std::vector<std::int8_t> h_interactions(count_all_interactions, 1);
  d_interactions.device_to_host(h_interactions);

  BOOST_CHECK_EQUAL_COLLECTIONS(
    h_interactions.begin(), h_interactions.end(), expected_interactions.begin(), expected_interactions.end());
}

BOOST_AUTO_TEST_SUITE_END()
