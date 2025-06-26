#define BOOST_TEST_MODULE wl_tests

#include <boost/test/unit_test.hpp>

#include "../src/interval.hpp"

BOOST_AUTO_TEST_SUITE(interval)

/**
 * Test typical 8x8 lattice with 16 intervals and 8 walkers with overlap 0.25
 */
BOOST_AUTO_TEST_CASE(test_typical_eight_times_eight_intervals)
{
  const int e_min = -128;
  const int e_max = 128;
  const unsigned int num_intervals = 16;
  const int num_walkers = 8;
  const float overlap = 0.25F;

  const int expected_len_interval = 54;
  const int expected_h_start_size = 16;
  const int expected_h_end_size = 16;
  const int expected_last_h_end = 128;
  const int expected_first_h_start = -128;
  const int expected_last_h_start = 67;
  const int expected_len_histogram_over_all_walkers = 62 * 8 + 54 * 8 * 15;

  wl::interval_results result{e_min, e_max, num_intervals, num_walkers, overlap};

  BOOST_CHECK_EQUAL(result.len_interval, expected_len_interval);
  BOOST_CHECK_EQUAL(result.h_start.size(), expected_h_start_size);
  BOOST_CHECK_EQUAL(result.h_end.size(), expected_h_end_size);
  BOOST_CHECK_EQUAL(result.h_end[15], expected_last_h_end);
  BOOST_CHECK_EQUAL(result.h_start[0], expected_first_h_start);
  BOOST_CHECK_EQUAL(result.h_start[15], expected_last_h_start);
  BOOST_CHECK_EQUAL(result.len_histogram_over_all_walkers, expected_len_histogram_over_all_walkers);
}

/**
 * Test of invalid argument error throw for empty interval scenario
 */
BOOST_AUTO_TEST_CASE(test_empty_interval_throw)
{
  const int e_min = -16;
  const int e_max = 16;
  const unsigned int num_intervals = 64;
  const int num_walkers = 8;
  const float overlap = 1;

  BOOST_CHECK_THROW(wl::interval_results(e_min, e_max, num_intervals, num_walkers, overlap), std::invalid_argument);
}

/**
 * Test of invalid argument error throw for out of bounds overlap
 */
BOOST_AUTO_TEST_CASE(test_out_of_bounds_overlap_throw)
{
  const int e_min = -16;
  const int e_max = 16;
  const unsigned int num_intervals = 64;
  const int num_walkers = 8;
  const float overlap = 1.1F;

  BOOST_CHECK_THROW(wl::interval_results(e_min, e_max, num_intervals, num_walkers, overlap), std::invalid_argument);
}

BOOST_AUTO_TEST_SUITE_END()
