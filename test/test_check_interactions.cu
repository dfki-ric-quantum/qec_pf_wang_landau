#include <boost/test/tools/old/interface.hpp>
#define BOOST_TEST_MODULE wl_tests

#include <vector>

#include <boost/test/unit_test.hpp>
#include <cuda_runtime.h>

#include "../src/cuda_utils.cuh"
#include "../src/wl.cuh"

BOOST_AUTO_TEST_SUITE(check_interactions_finished)

/**
 * Test that d_cond is set to -1 if all intervals are finished.
 */
BOOST_AUTO_TEST_CASE(check_all_finished)
{
  const int num_interactions = 100;
  const int num_intervals = 20;

  const int total_intervals = num_interactions * num_intervals;

  wl::device_tmp d_tmp_storage{};

  wl::device_ptr<int8_t> d_cond_interval(total_intervals, 1);
  wl::device_ptr<int> d_cond_interactions(num_interactions, 0);
  wl::device_ptr<int> d_offset_intervals(num_interactions + 1);

  std::vector<int> h_offset_intervals(num_interactions + 1);

  for (int i = 0; i < num_interactions; i++) {
    h_offset_intervals[i] = i * num_intervals;
  }

  h_offset_intervals[num_interactions] = total_intervals;

  d_offset_intervals.host_to_device(h_offset_intervals);

  wl::check_interactions_finished(d_cond_interactions.data(),
                                  d_tmp_storage,
                                  d_cond_interval.data(),
                                  d_offset_intervals.data(),
                                  num_intervals,
                                  num_interactions);

  std::vector<int> h_cond_interactions(num_interactions, 0);
  d_cond_interactions.device_to_host(h_cond_interactions);

  for (int i = 0; i < num_interactions; i++) {
    BOOST_CHECK(h_cond_interactions[i] == -1);
  }
}

/**
 * Sett all d_cond_intervals to up, except for last one to see whether d_cond_interaction is set correct.
 */
BOOST_AUTO_TEST_CASE(check_last_not_finished)
{
  const int num_interactions = 100;
  const int num_intervals = 20;

  const int total_intervals = num_interactions * num_intervals;

  wl::device_tmp d_tmp_storage{};

  std::vector<int8_t> h_cond_interval(total_intervals, 1);
  h_cond_interval[total_intervals - 1] = 0;

  wl::device_ptr<int8_t> d_cond_interval(total_intervals, 0);
  wl::device_ptr<int> d_cond_interactions(num_interactions, 0);
  wl::device_ptr<int> d_offset_intervals(num_interactions + 1);

  std::vector<int> h_offset_intervals(num_interactions + 1);

  for (int i = 0; i < num_interactions; i++) {
    h_offset_intervals[i] = i * num_intervals;
  }

  h_offset_intervals[num_interactions] = total_intervals;

  d_offset_intervals.host_to_device(h_offset_intervals);
  d_cond_interval.host_to_device(h_cond_interval);

  wl::check_interactions_finished(d_cond_interactions.data(),
                                  d_tmp_storage,
                                  d_cond_interval.data(),
                                  d_offset_intervals.data(),
                                  num_intervals,
                                  num_interactions);

  std::vector<int> h_cond_interactions(num_interactions, 0);
  d_cond_interactions.device_to_host(h_cond_interactions);

  for (int i = 0; i < num_interactions - 1; i++) {
    BOOST_CHECK(h_cond_interactions[i] == -1);
  }

  BOOST_CHECK(h_cond_interactions[num_interactions - 1] != -1);
}

/**
 * Test no interaction is finished if all intervals are not finished
 */
BOOST_AUTO_TEST_CASE(check_all_not_finished)
{
  const int num_interactions = 100;
  const int num_intervals = 20;

  const int total_intervals = num_interactions * num_intervals;

  wl::device_tmp d_tmp_storage{};

  wl::device_ptr<int8_t> d_cond_interval(total_intervals, 0);
  wl::device_ptr<int> d_cond_interactions(num_interactions, 0);
  wl::device_ptr<int> d_offset_intervals(num_interactions + 1);

  std::vector<int> h_offset_intervals(num_interactions + 1);

  for (int i = 0; i < num_interactions; i++) {
    h_offset_intervals[i] = i * num_intervals;
  }

  h_offset_intervals[num_interactions] = total_intervals;

  d_offset_intervals.host_to_device(h_offset_intervals);

  wl::check_interactions_finished(d_cond_interactions.data(),
                                  d_tmp_storage,
                                  d_cond_interval.data(),
                                  d_offset_intervals.data(),
                                  num_intervals,
                                  num_interactions);

  std::vector<int> h_cond_interactions(num_interactions, 0);
  d_cond_interactions.device_to_host(h_cond_interactions);

  for (int i = 0; i < num_interactions - 1; i++) {
    BOOST_CHECK(h_cond_interactions[i] != -1);
  }
}

BOOST_AUTO_TEST_SUITE_END()
