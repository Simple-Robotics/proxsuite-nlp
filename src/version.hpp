#pragma once

#include "proxnlp/config.hpp"

#include <string>
#include <sstream>

namespace proxnlp {
/// @brief    Pretty-print the package version number.
/// @param    delimiter   The delimiter between the major/minor/patch version
/// components.
inline std::string printVersion(const std::string &delimiter = ".") {
  std::ostringstream oss;
  oss << PROXNLP_MAJOR_VERSION << delimiter << PROXNLP_MINOR_VERSION
      << delimiter << PROXNLP_PATCH_VERSION;
  return oss.str();
}

} // namespace proxnlp
