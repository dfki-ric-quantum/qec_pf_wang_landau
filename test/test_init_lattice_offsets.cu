#define BOOST_TEST_MODULE wl_tests

#include <vector>

#include <boost/test/unit_test.hpp>
#include <cuda_runtime.h>

#include "../src/cuda_utils.cuh"
#include "../src/wl.cuh"

BOOST_AUTO_TEST_SUITE(init_lattice_offsets)

/**
 * Test initialization of lattice offsets for dim 4x4 over 10 lattices.
 */
BOOST_AUTO_TEST_CASE(test_init_offsets_lattice)
{
    const int lattice_dim_x = 4;
    const int lattice_dim_y = 4;
    const int num_lattices = 10;
    const int count_all_offsets = num_lattices;

    std::vector<int> expected_offsets(count_all_offsets);
    for (int i = 0; i < num_lattices; ++i) {
        expected_offsets[i] = i * lattice_dim_x * lattice_dim_y;
    }

    wl::device_ptr<int> d_offset_lattice(num_lattices);

    const int THREADS = 128;
    const int BLOCKS = (count_all_offsets + THREADS - 1) / THREADS;

    wl::init_offsets_lattice<<<BLOCKS, THREADS>>>(d_offset_lattice.data(), lattice_dim_x, lattice_dim_y, count_all_offsets);

    std::vector<int> h_offset_lattice(count_all_offsets);
    cudaMemcpy(h_offset_lattice.data(), d_offset_lattice.data(), count_all_offsets * sizeof(int), cudaMemcpyDeviceToHost);

    BOOST_CHECK_EQUAL_COLLECTIONS(
      h_offset_lattice.begin(), h_offset_lattice.end(),
      expected_offsets.begin(), expected_offsets.end()
    );
}

BOOST_AUTO_TEST_SUITE_END()