#include "lienlp/python/fwd.hpp"
#include "lienlp/problem-base.hpp"


namespace lienlp {
  namespace python {
    
    void exposeProblem()
    {
      using context::Problem_t;
      bp::class_<Problem_t, std::shared_ptr<Problem_t>>(
        "Problem", "Problem definition class.",
        bp::init<const context::Cost_t&>(bp::args("cost"))
      )
        ;
    }

  } // namespace python
} // namespace lienlp

