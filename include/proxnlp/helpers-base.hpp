#pragma once

#include "proxnlp/fwd.hpp"

namespace proxnlp {
namespace helpers {

template <typename Scalar> struct base_callback {
  virtual void call(const WorkspaceTpl<Scalar> &,
                    const ResultsTpl<Scalar> &) = 0;
  virtual ~base_callback() = default;
};

} // namespace helpers
} // namespace proxnlp
