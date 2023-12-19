/// @file
/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#pragma once

#include "proxsuite-nlp/config.hpp"

#include <string>
#include <sstream>

namespace proxsuite {
namespace nlp {
/// @brief    Pretty-print the package version number.
/// @param    delimiter   The delimiter between the major/minor/patch version
/// components.
inline std::string printVersion(const std::string &delimiter = ".") {
  std::ostringstream oss;
  oss << PROXSUITE_NLP_MAJOR_VERSION << delimiter << PROXSUITE_NLP_MINOR_VERSION
      << delimiter << PROXSUITE_NLP_PATCH_VERSION;
  return oss.str();
}

} // namespace nlp
} // namespace proxsuite
