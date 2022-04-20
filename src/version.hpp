#pragma once

#include "lienlp/config.hpp"

#include <string>
#include <sstream>

namespace lienlp
{
  /// @brief    Pretty-print the package version number.
  /// @param    delimiter   The delimiter between the major/minor/patch version components.
  inline std::string printVersion(const std::string& delimiter = ".")
  {
    std::ostringstream oss;
    oss << LIENLP_MAJOR_VERSION << delimiter
        << LIENLP_MINOR_VERSION << delimiter
        << LIENLP_PATCH_VERSION;
    return oss.str();
  }
  
} // namespace lienlp

