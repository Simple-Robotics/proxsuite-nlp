#include "lienlp/python/fwd.hpp"

#include "lienlp/workspace.hpp"


namespace lienlp
{
  namespace python
  {
    void exposeWorkspace()
    {
      using context::Scalar;
      bp::class_<SWorkspace<Scalar>>(
        "Workspace", "Solver workspace.",
        bp::init<int, int, const context::Problem_t&>(bp::args("nx", "ndx", "problem"))
      );
    }
    
  } // namespace python
} // namespace lienlp

