/// @file
/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#pragma once

#include "proxsuite-nlp/helpers-base.hpp"

namespace proxsuite {
namespace nlp {
namespace helpers {

template <typename Scalar> struct print_callback : base_callback<Scalar> {
  void call(const WorkspaceTpl<Scalar> &workspace,
            const ResultsTpl<Scalar> &results) {}
};

} // namespace helpers
} // namespace nlp
} // namespace proxsuite
