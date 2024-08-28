#pragma once

#include "proxsuite-nlp/context.hpp"
#include <eigenpy/eigenpy.hpp>

namespace proxsuite {
namespace nlp {

/// @brief Python bindings.
namespace python {
namespace bp = boost::python;

/// User-defined literal for bp::arg
inline bp::arg operator""_a(const char *argname, std::size_t) {
  return bp::arg(argname);
}

void exposeFunctionTypes();
void exposeManifolds();
/// Expose defined residuals for modelling
void exposeResiduals();
#ifdef PROXSUITE_NLP_WITH_PINOCCHIO
/// Expose residuals dependent on Pinocchio
void exposePinocchioResiduals();
#endif
void exposeCost();
void exposeConstraints();
void exposeProblem();
void exposeResults();
void exposeWorkspace();
void exposeLdltRoutines();
void exposeSolver();
void exposeCallbacks();
void exposeAutodiff();

} // namespace python

} // namespace nlp
} // namespace proxsuite
