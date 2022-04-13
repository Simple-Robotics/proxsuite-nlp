#include "lienlp/python/fwd.hpp"

#include "lienlp/workspace.hpp"


namespace lienlp
{
  namespace python
  {
    void exposeWorkspace()
    {
      using context::Scalar;
      bp::class_<context::Workspace>(
        "Workspace", "SolverTpl workspace.",
        bp::init<int, int, const context::Problem&>(bp::args("nx", "ndx", "problem"))
      )
        .def_readonly("kkt_matrix", &context::Workspace::kktMatrix, "KKT matrix buffer.")
        .def_readonly("kkt_rhs", &context::Workspace::kktRhs, "KKT system right-hand side buffer.")
        .def_readonly("primal_residuals", &context::Workspace::primalResiduals, "Vector constraint residuals.")
        .def_readonly("dual_residuals", &context::Workspace::dualResidual, "Dual vector residual.")
      ;
    }
    
  } // namespace python
} // namespace lienlp

