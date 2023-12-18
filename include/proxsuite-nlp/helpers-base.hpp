/// @file
/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#pragma once

#include "proxsuite-nlp/fwd.hpp"

namespace proxsuite {
namespace nlp {
namespace helpers {

template <typename Scalar> struct base_callback {
  virtual void call(const WorkspaceTpl<Scalar> &,
                    const ResultsTpl<Scalar> &) = 0;
  virtual ~base_callback() = default;
};

} // namespace helpers
} // namespace nlp
} // namespace proxsuite
