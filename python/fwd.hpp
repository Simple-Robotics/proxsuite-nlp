#pragma once

#include <eigenpy/eigenpy.hpp>

#ifdef byte
#undef byte
#endif
#include "proxnlp/context.hpp"

namespace proxnlp {

/// @brief Python bindings.
namespace python {
namespace bp = boost::python;

void exposeFunctionTypes();
void exposeManifolds();
/// Expose defined residuals for modelling
void exposeResiduals();
/// Expose residuals dependent on Pinocchio
void exposePinocchioResiduals();
void exposeCost();
void exposeConstraints();
void exposeProblem();
void exposeResults();
void exposeWorkspace();
void exposeSolver();
void exposeCallbacks();
void exposeAutodiff();

} // namespace python

} // namespace proxnlp
