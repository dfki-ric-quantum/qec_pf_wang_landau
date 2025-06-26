#define BOOST_TEST_MODULE wl_tests

#include <array>
#include <numeric>
#include <stdexcept>

#include "../src/cuda_utils.cuh"
#include <boost/test/unit_test.hpp>
#include <cuda_runtime.h>

// NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic)
namespace {
/**
 * Helper kernel to multiply all values in a device pointer with a scalar.
 *
 * @param dst the destination pointer
 * @param src the source pointer
 * @param len number of elements to operate on
 * @param scale the scalar to multiply with
 */
__global__ void multiply(int* dst, const int* src, std::size_t len, int scale)
{
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

  while (tid < len) {
    dst[tid] = src[tid] * scale;
    tid += gridDim.x * blockDim.x;
  }
}
}
// NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic)

/**
 * Test the device pointer abstraction.
 */
BOOST_AUTO_TEST_SUITE(device_ptr)

/**
 * Sanity check, if we can create a pointer and it is properly initialized.
 */
BOOST_AUTO_TEST_CASE(empty_ptr)
{
  const std::size_t size = 10;

  wl::device_ptr<float> d_test{size};
  BOOST_CHECK_EQUAL(d_test.size(), size);
}

/**
 * Test if a device pointer can be created from a host range.
 *
 * The test checks if data is correctly copied to the device and back.
 */
BOOST_AUTO_TEST_CASE(init_from_range)
{
  const std::size_t size = 256;
  const int value = 10;

  std::vector src(size, value);
  std::vector dst(size, 0);

  wl::device_ptr<int> d_test{src};
  d_test.device_to_host(dst);

  BOOST_CHECK_EQUAL(d_test.size(), size);
  BOOST_CHECK_EQUAL(dst[0], value);
  BOOST_CHECK(src == dst);
}

/**
 * Test kernel call with simple arithmetic operation on device_ptr.
 *
 * The test checks if the operation on the device pointer leads to the same result as on a host
 * range.
 */
BOOST_AUTO_TEST_CASE(kernel_call)
{
  const unsigned int PAR = 16;
  const std::size_t size = 256;
  const int scale = 42;

  std::vector src(size, 0);
  std::vector dst(size, 0);

  std::iota(src.begin(), src.end(), 1);

  wl::device_ptr<int> d_src{src};
  wl::device_ptr<int> d_dst{dst};

  multiply<<<PAR, PAR>>>(d_dst.data(), d_src.data(), d_src.size(), scale);

  d_dst.device_to_host(dst);

  std::ranges::transform(src, src.begin(), [](auto elem) { return scale * elem; });
  BOOST_CHECK(src == dst);
}

/**
 * Test single element access
 */
BOOST_AUTO_TEST_CASE(element_access)
{
  const std::size_t size = 256;
  std::array<int, size> h_data{};

  std::iota(h_data.begin(), h_data.end(), 0);
  wl::device_ptr<int> d_data{h_data};

  std::size_t idx = 0;

  BOOST_CHECK(std::ranges::all_of(h_data, [&idx, &d_data](auto elem) {
    bool equal = (elem == d_data[idx]);
    ++idx;
    return equal;
  }));
}

/**
 * Test read from device pointer with offset and size
 */
BOOST_AUTO_TEST_CASE(offset_access)
{
  const std::size_t size = 256;
  const std::size_t partial_size = 20;
  const std::size_t offset = 100;

  std::array<int, size> h_data{};
  std::array<int, partial_size> h_res{};

  std::iota(h_data.begin(), h_data.end(), 0);
  wl::device_ptr<int> d_data{h_data};

  d_data.device_to_host(h_res, offset, partial_size);

  std::size_t idx = offset;

  BOOST_CHECK(std::ranges::all_of(h_res, [&idx, &h_data](auto elem) {
    bool equal = (elem == h_data.at(idx));
    ++idx;
    return equal;
  }));
}

/**
 * Test that out of bound copy from host to device throws
 */
BOOST_AUTO_TEST_CASE(oob_host_device)
{
  const std::size_t device_size = 256;
  const std::size_t host_size = 128;

  std::array<int, host_size> h_data{};
  std::iota(h_data.begin(), h_data.end(), 0);

  wl::device_ptr<int> d_data(device_size, 0);

  BOOST_REQUIRE_THROW(d_data.host_to_device(h_data), std::out_of_range);
}

/**
 * Test that out of bound copy from device to host throws
 */
BOOST_AUTO_TEST_CASE(oob_device_host)
{
  const std::size_t device_size = 256;
  const std::size_t host_size = 128;

  std::array<int, host_size> h_data{};
  wl::device_ptr<int> d_data(device_size, 0);

  BOOST_REQUIRE_THROW(d_data.device_to_host(h_data), std::out_of_range);
}

/**
 * Test that of bound copy from device to host with wrong offset throws
 */
BOOST_AUTO_TEST_CASE(oob_device_host_wrong_offset)
{
  const std::size_t device_size = 256;
  const std::size_t host_size = 128;
  const std::size_t offset = 512;

  std::array<int, host_size> h_data{};
  wl::device_ptr<int> d_data(device_size, 0);

  BOOST_REQUIRE_THROW(d_data.device_to_host(h_data, offset, host_size), std::out_of_range);
}

/**
 * Test that of bound copy from device to host with too small host buffer throws
 */
BOOST_AUTO_TEST_CASE(oob_device_host_dst_too_small)
{
  const std::size_t device_size = 256;
  const std::size_t host_size = 128;
  const std::size_t offset = 10;

  std::array<int, host_size> h_data{};
  wl::device_ptr<int> d_data(device_size, 0);

  BOOST_REQUIRE_THROW(d_data.device_to_host(h_data, offset, host_size + 10), std::out_of_range);
}

/**
 * Test that of bound copy from device to host beyond device memory throws
 */
BOOST_AUTO_TEST_CASE(oob_device_host_beyond_db)
{
  const std::size_t device_size = 256;
  const std::size_t host_size = 128;
  const std::size_t offset = 200;

  std::array<int, host_size> h_data{};
  wl::device_ptr<int> d_data(device_size, 0);

  BOOST_REQUIRE_THROW(d_data.device_to_host(h_data, offset, host_size), std::out_of_range);
}

BOOST_AUTO_TEST_SUITE_END()
