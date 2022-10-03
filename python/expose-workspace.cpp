#include "proxnlp/python/fwd.hpp"

#include "proxnlp/workspace.hpp"

namespace proxnlp {
namespace python {
void exposeWorkspace() {
  using context::Scalar;
  bp::class_<context::Workspace>("Workspace", "SolverTpl workspace.",
                                 bp::init<int, int, const context::Problem &>(
                                     bp::args("self", "nx", "ndx", "problem")))
      .def_readonly("kkt_matrix", &context::Workspace::kkt_matrix,
                    "KKT matrix buffer.")
      .def_readonly("kkt_rhs", &context::Workspace::kkt_rhs,
                    "KKT system right-hand side buffer.")
      .def_readonly("cstr_values", &context::Workspace::cstr_values,
                    "Vector constraint residuals.")
      .def_readonly("cstr_values_proj", &context::Workspace::cstr_values_proj,
                    "Projected constraint residuals.")
      .def_readonly("dual_residuals", &context::Workspace::dual_residual,
                    "Dual vector residual.")
      .def_readonly("jacobians_data", &context::Workspace::jacobians_data)
      .def_readonly("hessians_data", &context::Workspace::hessians_data)
      .def_readonly("cstr_jacobians", &context::Workspace::cstr_jacobians,
                    "Block jacobians.")
      .def_readonly("lams_plus", &context::Workspace::lams_plus,
                    "First-order multiplier estimates.")
      .def_readonly("lams_pdal", &context::Workspace::lams_pdal,
                    "Primal-dual multiplier estimates.");
}

} // namespace python
} // namespace proxnlp
