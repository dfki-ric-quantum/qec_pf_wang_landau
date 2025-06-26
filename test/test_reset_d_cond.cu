#define BOOST_TEST_MODULE wl_tests

#include <algorithm>
#include <cmath>
#include <vector>

#include <boost/test/unit_test.hpp>
#include <cuda_runtime.h>

#include "../src/cuda_utils.cuh"
#include "../src/wl.cuh"


/**
 * Test wl-convergence condition flag reset
 */
BOOST_AUTO_TEST_SUITE(reset_cond_d)

/**
 * Test that no reset happens, if all converged
 */
BOOST_AUTO_TEST_CASE(reste_none)
{
  const double beta = 1e-10;

  const int num_interactions = 2;
  const int num_intervals = 10;
  const int walker_per_interval = 20;

  const int total_intervals = num_interactions * num_intervals;
  const int total_walker = num_interactions * num_intervals * walker_per_interval;

  std::vector<std::int8_t> h_cond(total_intervals, 1);
  std::vector<double> h_factor(total_walker, std::exp(beta));

  wl::device_ptr<std::int8_t> d_cond(h_cond);
  wl::device_ptr<double> d_factor(h_factor);

  wl::reset_d_cond<<<num_interactions, num_intervals>>>(
    d_cond.data(), d_factor.data(), total_intervals, beta, walker_per_interval);

  d_cond.device_to_host(h_cond);

  BOOST_CHECK(std::ranges::all_of(h_cond, [](auto elem) { return elem == 1; }));
}

/**
 * Test that reset happens for all, if none converged
 */
BOOST_AUTO_TEST_CASE(reste_all)
{
  const double beta = 1e-10;
  const double diff = 1e-3;

  const int num_interactions = 2;
  const int num_intervals = 10;
  const int walker_per_interval = 20;

  const int total_intervals = num_interactions * num_intervals;
  const int total_walker = num_interactions * num_intervals * walker_per_interval;

  std::vector<std::int8_t> h_cond(total_intervals, 1);
  std::vector<double> h_factor(total_walker, std::exp(beta) + diff);

  wl::device_ptr<std::int8_t> d_cond(h_cond);
  wl::device_ptr<double> d_factor(h_factor);

  wl::reset_d_cond<<<num_interactions, num_intervals>>>(
    d_cond.data(), d_factor.data(), total_intervals, beta, walker_per_interval);

  d_cond.device_to_host(h_cond);

  BOOST_CHECK(std::ranges::all_of(h_cond, [](auto elem) { return elem == 0; }));
}

/**
 * Test that only interactions get reset, that did not converge
 */
BOOST_AUTO_TEST_CASE(reste_some)
{
  const double beta = 1e-10;
  const double diff = 1e-3;

  const int num_interactions = 2;
  const int num_intervals = 10;
  const int walker_per_interval = 20;

  const int total_intervals = num_interactions * num_intervals;
  const int total_walker = num_interactions * num_intervals * walker_per_interval;

  const std::size_t factor_offset = total_walker - (num_intervals * walker_per_interval);
  const std::size_t cond_offset = total_intervals - num_intervals;

  std::vector<std::int8_t> h_cond(total_intervals, 1);
  std::vector<double> h_factor(total_walker, std::exp(beta));

  // Add diff to last interaction
  auto factor_offset_it = h_factor.begin() + factor_offset;
  std::ranges::transform(factor_offset_it, h_factor.end(), factor_offset_it, [diff](auto elem) { return elem + diff; });

  wl::device_ptr<std::int8_t> d_cond(h_cond);
  wl::device_ptr<double> d_factor(h_factor);

  wl::reset_d_cond<<<num_interactions, num_intervals>>>(
    d_cond.data(), d_factor.data(), total_intervals, beta, walker_per_interval);

  d_cond.device_to_host(h_cond);

  auto cond_offset_it = h_cond.begin() + cond_offset;
  BOOST_CHECK(std::ranges::all_of(h_cond.begin(), cond_offset_it, [](auto elem) { return elem == 1; }));
  BOOST_CHECK(std::ranges::all_of(cond_offset_it, h_cond.end(), [](auto elem) { return elem == 0; }));
}

BOOST_AUTO_TEST_SUITE_END()
