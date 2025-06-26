#ifndef BOUNDARY_HPP_
#define BOUNDARY_HPP_

#include <format>
#include <stdexcept>
#include <string>
#include <string_view>

namespace wl {
enum class boundary : unsigned int
{
  periodic = 0,
  open = 1
};

inline std::string boundary_to_str(wl::boundary boundary)
{
  if (boundary == wl::boundary::periodic) {
    return "periodic";
  }
  if (boundary == wl::boundary::open) {
    return "open";
  }
  throw std::runtime_error("Unsupported boundary condition");
}

inline wl::boundary str_to_boundary(std::string_view boundary)
{
  if (boundary == "periodic") {
    return wl::boundary::periodic;
  }
  if (boundary == "open") {
    return wl::boundary::open;
  }

  throw std::runtime_error(std::format("Unsupported boundary condition {}", boundary));
}

} // wl

#endif // BOUNDARY_HPP_
