#pragma once

#include <pinocchio/fwd.hpp>
#include "proxnlp/python/fwd.hpp"
#include "proxnlp/manifold-base.hpp"
#include "proxnlp/modelling/spaces/cartesian-product.hpp"

namespace proxnlp {
namespace python {

namespace internal {

/// Expose the base manifold type
void exposeManifoldBase();
} // namespace internal

} // namespace python

} // namespace proxnlp
