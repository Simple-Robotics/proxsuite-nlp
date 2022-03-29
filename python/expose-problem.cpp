#include "lienlp/python/fwd.hpp"
#include "lienlp/problem-base.hpp"


#include <boost/python/suite/indexing/vector_indexing_suite.hpp>


namespace lienlp
{
  namespace python
  {
    
    void exposeProblem()
    {
      using context::Problem_t;
      using ConstraintPtr = shared_ptr<context::Constraint_t>;
      bp::class_<Problem_t, shared_ptr<Problem_t>>(
        "Problem", "Problem definition class.",
        bp::init<const context::Cost_t&,
                 const std::vector<ConstraintPtr>&
                 >(bp::args("cost", "constraints"))
      )
        .def(bp::init<const context::Cost_t&>(bp::args("cost")))
        .add_property("num_constraints", &Problem_t::getNumConstraints)
        .add_property("total_constraint_dim", &Problem_t::getTotalConstraintDim)
        .add_property("constraint_dims", &Problem_t::getConstraintDims)
        .def("add_constraint", &Problem_t::addConstraint, bp::args("cstr"),
             "Add a constraint to the problem.")
        ;

    }

  } // namespace python
} // namespace lienlp

