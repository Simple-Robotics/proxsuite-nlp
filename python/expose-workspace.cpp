#include "lienlp/python/fwd.hpp"

#include "lienlp/workspace.hpp"


namespace lienlp
{
  namespace python
  {
    void exposeWorkspace()
    {
      using context::Scalar;
      bp::class_<context::Workspace_t>(
        "Workspace", "SolverTpl workspace.",
        bp::init<int, int, const context::Problem_t&>(bp::args("nx", "ndx", "problem"))
      )
        .def_readonly("kkt_matrix", &context::Workspace_t::kktMatrix, "KKT matrix buffer.")
        .def_readonly("kkt_rhs", &context::Workspace_t::kktRhs, "KKT system right-hand side buffer.")
      ;
    }
    
  } // namespace python
} // namespace lienlp

