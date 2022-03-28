#include "lienlp/python/fwd.hpp"
#include "lienlp/problem-base.hpp"


namespace lienlp
{
  namespace python
  {
    
    void exposeProblem()
    {
      using context::Problem_t;
      using ConstraintStack = std::vector<Problem_t::ConstraintPtr>;
      bp::class_<Problem_t, shared_ptr<Problem_t>>(
        "Problem", "Problem definition class.",
        bp::init<const context::Cost_t&,
                 ConstraintStack&
                 >(bp::args("cost", "constraints"))
      )
        .def(bp::init<const context::Cost_t&>(bp::args("cost")))
        ;
    }

  } // namespace python
} // namespace lienlp

