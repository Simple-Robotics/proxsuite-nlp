/// @file
/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#pragma once

#include "proxnlp/helpers-base.hpp"

namespace proxnlp {
namespace helpers {

template <typename Scalar> struct print_callback : base_callback<Scalar> {
  void call(const WorkspaceTpl<Scalar> &workspace,
            const ResultsTpl<Scalar> &results) {}
};

} // namespace helpers
} // namespace proxnlp
