#include "proxnlp/python/fwd.hpp"

#include "proxnlp/workspace.hpp"

namespace proxnlp {
namespace python {
void exposeWorkspace() {
  using context::Scalar;
  using context::Workspace;
  bp::class_<Workspace, boost::noncopyable>(
      "Workspace", "SolverTpl workspace.",
      bp::init<const context::Problem &>(bp::args("self", "problem")))
      .def_readonly("kkt_matrix", &Workspace::kkt_matrix, "KKT matrix buffer.")
      .def_readonly("kkt_rhs", &Workspace::kkt_rhs,
                    "KKT system right-hand side buffer.")
      .def_readonly("data_cstr_values", &Workspace::data_cstr_values)
      .def_readonly("cstr_values", &Workspace::cstr_values,
                    "Vector constraint residuals.")
      .def_readonly("data_shift_cstr_values",
                    &Workspace::data_shift_cstr_values,
                    "Shifted constraint values.")
      .def_readonly("shift_cstr_proj", &Workspace::shift_cstr_proj,
                    "Projected shifted constraint residuals.")
      .def_readonly("dual_residuals", &Workspace::dual_residual,
                    "Dual vector residual.")
      .def_readonly("data_jacobians", &Workspace::data_jacobians,
                    "Constraint Jacobians.")
      .def_readonly("data_hessians", &Workspace::data_hessians,
                    "Constraint vector-Hessian product matrices.")
      .def_readonly("cstr_jacobians", &Workspace::cstr_jacobians,
                    "Block jacobians.")
      .def_readonly("data_jacobians_proj", &Workspace::data_jacobians_proj,
                    "Projected constraint Jacobians.")
      .def_readonly("lams_plus", &Workspace::lams_plus,
                    "First-order multiplier estimates.")
      .def_readonly("lams_pdal", &Workspace::lams_pdal,
                    "Primal-dual multiplier estimates.")
      .def_readonly("alpha_opt", &Workspace::alpha_opt,
                    "Computed linesearch step length.")
      .def_readonly("dmerit_dir", &Workspace::dmerit_dir)
#ifdef PROXNLP_CUSTOM_LDLT
      .def(
          "ldlt",
          +[](const Workspace &ws) -> const linalg::ldlt_base<Scalar> & {
            return *ws.ldlt_;
          },
          bp::return_internal_reference<>(),
          "Returns a reference to the underlying LDLT solver.")
#endif
      ;

#ifdef PROXNLP_CUSTOM_LDLT
  using LDLTBase = linalg::ldlt_base<Scalar>;
  bp::class_<LDLTBase, boost::noncopyable>(
      "LDLTBase", "Base class for LDLT solvers.", bp::no_init)
      .def("matrixLDLT", &LDLTBase::matrixLDLT,
           bp::return_value_policy<bp::return_by_value>(),
           "Get the current value of the decomposition matrix. This makes a "
           "copy.");
#endif
}

} // namespace python
} // namespace proxnlp
