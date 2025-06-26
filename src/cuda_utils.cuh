#ifndef CUDA_UTILS_CUH_
#define CUDA_UTILS_CUH_

#include <format>
#include <ranges>
#include <source_location>
#include <stdexcept>
#include <type_traits>

#include <cuda_runtime.h>

namespace wl {

/**
 * Handle CUDA errors
 * Checks for success of a cuda call and throws a std::runtime_error on error
 *
 * @param error The error code returned from the cuda call
 * @param loc The source location, defaults to the place the handler was called
 * from
 */
inline void handle_cuda(cudaError_t error, std::source_location loc = std::source_location::current())
{
  if (error != cudaSuccess) {
    throw std::runtime_error{
      std::format("Cuda error in file '{}', line {} : {}", loc.file_name(), loc.line(), cudaGetErrorString(error))};
  }
}

/**
 * RAII wrapper for temporary device storage needed by some API calls (e.g. CUB)
 */
template<typename T = void>
class device_tmp
{
  T* ptr_ = nullptr;
  std::size_t count_{};

  void _allocate()
  {
    if constexpr (std::is_void_v<T>) {
      handle_cuda(cudaMalloc((void**)(&ptr_), count_));
    } else {
      handle_cuda(cudaMalloc((void**)(&ptr_), sizeof(T) * count_));
    }
  }

public:
  /**
   * The default constructor, creates an unallocated pointer.
   */
  device_tmp() = default;
  device_tmp(const device_tmp<T>&) = delete;
  device_tmp(device_tmp<T>&&) = delete;

  /**
   * Construtor to initialize empty pointer.
   * Allocates `count` elements of `T`. Throws std::runtime_error on
   * failure
   *
   * @param count Number of elements of `T` to allocate for.
   */
  explicit device_tmp(std::size_t count)
    : count_(count)
  {
    _allocate();
  }

  void operator=(const device_tmp<T>&) = delete;
  void operator=(device_tmp<T>&&) = delete;

  ~device_tmp() { cudaFree(ptr_); }

  /**
   * Resize allocated memory if needed.
   *
   * If the currently allocated memory is smaller than requested, it is freed and new
   * memory of appropriate size allocated. If the already allocated memory is sufficiently
   * large, this method does nothing. Note: Old data is _not_ copied to the new
   * storage.
   *
   * @param count Number of elements of `T` to allocate
   */
  void resize(std::size_t count)
  {
    if (count_ >= count) {
      return;
    }

    if (ptr_ != nullptr) {
      handle_cuda(cudaFree(ptr_));
    }

    count_ = count;
    _allocate();
  }

  /**
   * Return the wrapped pointer.
   */
  [[nodiscard]] T* data() { return ptr_; }
  /**
   * Return the number of allocated elements.
   */
  [[nodiscard]] std::size_t size() const { return count_; }
};

/**
 * RAII wrapper for CUDA device pointers
 * Currently can't be default contructed, copied or moved.
 */
template<typename T>
class device_ptr
{
  T* ptr_;
  std::size_t count_;

  void _allocate() { handle_cuda(cudaMalloc((void**)(&ptr_), sizeof(T) * count_)); }

public:
  device_ptr() = delete;
  device_ptr(const device_ptr<T>&) = delete;
  device_ptr(device_ptr<T>&&) = delete;

  /**
   * Construtor to initialize empty pointer.
   * Allocates `count` elements of `T`. Throws std::runtime_error on
   * failure
   *
   * @param count Number of elements of `T` to allocate for.
   */
  explicit device_ptr(std::size_t count)
    : count_(count)
  {
    _allocate();
  }

  /**
   * Construtor to initialize pointer filled with provided value.
   * Allocates `count` elements of `T` and write the provided value to each byte.
   * Throws std::runtime_error on failure
   *
   * @param count Number of elements of `T` to allocate for.
   * @param value The value to set each byte to
   */
  device_ptr(std::size_t count, int value)
    : count_(count)
  {
    _allocate();
    fill(value);
  }

  /**
   * Construtor to initialize device pointer from host range.
   *
   * Allocates sufficient memory to store the content of the host range
   * and copies it to the device pointer.
   *
   * @param src The host range to copy
   */
  template<std::ranges::contiguous_range R>
    requires(std::same_as<T, std::ranges::range_value_t<R>>)
  explicit device_ptr(const R& src)
    : count_(std::ranges::size(src))
  {
    _allocate();
    host_to_device(src);
  }

  void operator=(const device_ptr<T>&) = delete;
  void operator=(device_ptr<T>&&) = delete;

  ~device_ptr() { cudaFree(ptr_); }

  /**
   * Copy host memory to device pointer
   *
   * @param src The source range on the host. Must be contiguous and have T as
   * value type.
   */
  template<std::ranges::contiguous_range R>
    requires(std::same_as<T, std::ranges::range_value_t<R>>)
  void host_to_device(const R& src)
  {
    if (std::ranges::size(src) != count_) {
      throw std::out_of_range(std::format(
        "Attempted to copy host buffer of size {} to device buffer of size {}", std::ranges::size(src), count_));
    }
    handle_cuda(cudaMemcpy(ptr_, std::ranges::data(src), sizeof(T) * count_, cudaMemcpyHostToDevice));
  }

  /**
   * Copy device pointer to host range
   *
   * @param dst The destination range on the host. Must be contiguous and have T
   * as value type.
   */
  template<std::ranges::contiguous_range R>
    requires(std::same_as<T, std::ranges::range_value_t<R>>)
  void device_to_host(R& dst) const
  {
    if (std::ranges::size(dst) != count_) {
      throw std::out_of_range(std::format(
        "Attempted to copy device buffer of size {} to host buffer of size {}", count_, std::ranges::size(dst)));
    }
    handle_cuda(cudaMemcpy(std::ranges::data(dst), ptr_, sizeof(T) * count_, cudaMemcpyDeviceToHost));
  }

  /**
   * Copy `size` elements from device pointer to host, starting from offset
   *
   * @param dst The destination range on the host. Must be contiguous and have T
   * as value type.
   */
  template<std::ranges::contiguous_range R>
    requires(std::same_as<T, std::ranges::range_value_t<R>>)
  void device_to_host(R& dst, std::size_t offset, std::size_t size) const
  {
    bool dst_too_small = std::ranges::size(dst) < size;
    bool out_of_bound = (offset + size) > count_;

    if (dst_too_small || out_of_bound) {
      throw std::out_of_range(std::format(
        "Attempted to copy {} elements with offset {} from a device buffer of size {} to a host buffer of size {}",
        size,
        offset,
        count_,
        std::ranges::size(dst)));
    }

    handle_cuda(cudaMemcpy(std::ranges::data(dst), ptr_ + offset, sizeof(T) * size, cudaMemcpyDeviceToHost));
  }

  /**
   * Get a single element from the device memory to the host.
   *
   * NOTE: This does not perform bound checks
   */
  [[nodiscard]] T operator[](std::size_t idx) const
  {
    if (idx >= count_) {
      throw std::out_of_range(std::format("Access to device buffer of size {} with index {}.", count_, idx));
    }

    T res{};
    handle_cuda(cudaMemcpy(&res, &ptr_[idx], sizeof(T), cudaMemcpyDeviceToHost));
    return res;
  }

  /**
   * Fill device pointer bytewise with specified value
   *
   * @param val The value to set each byte to
   */
  void fill(int val) { handle_cuda(cudaMemset(ptr_, val, sizeof(T) * count_)); }

  /**
   * Return the wrapped pointer.
   */
  [[nodiscard]] T* data() { return ptr_; }
  /**
   * Return the number of allocated elements.
   */
  [[nodiscard]] std::size_t size() const { return count_; }
};

} // namespace wl

#endif // CUDA_UTILS_CUH_
