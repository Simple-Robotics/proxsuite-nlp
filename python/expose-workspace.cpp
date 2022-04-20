#include "proxnlp/python/fwd.hpp"

#include "proxnlp/workspace.hpp"


namespace proxnlp
{
  namespace python
  {
    void exposeWorkspace()
    {
      using context::Scalar;
      bp::class_<context::Workspace>(
        "Workspace", "SolverTpl workspace.",
        bp::init<int, int, const context::Problem&>(bp::args("self", "nx", "ndx", "problem"))
      )
        .def_readonly("kkt_matrix", &context::Workspace::kktMatrix, "KKT matrix buffer.")
        .def_readonly("kkt_rhs", &context::Workspace::kktRhs, "KKT system right-hand side buffer.")
        .def_readonly("primal_residuals", &context::Workspace::primalResiduals, "Vector constraint residuals.")
        .def_readonly("dual_residuals", &context::Workspace::dualResidual, "Dual vector residual.")
        .def_readonly("jacobians_data", &context::Workspace::jacobians_data)
        .def_readonly("hessians_data", &context::Workspace::hessians_data)
        .def_readonly("lams_plus", &context::Workspace::lamsPlus, "First-order multiplier estimates.")
        .def_readonly("lams_pdal",   &context::Workspace::lamsPDAL, "Primal-dual multiplier estimates.")
      ;
    }
    
  } // namespace python
} // namespace proxnlp

