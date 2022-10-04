#include "proxnlp/python/fwd.hpp"
#include "proxnlp/problem-base.hpp"

namespace proxnlp {
namespace python {

void exposeProblem() {
  using context::Constraint;
  using context::Manifold;
  using context::Problem;

  using CostPtr = shared_ptr<context::Cost>;

  bp::class_<Problem, shared_ptr<Problem>>(
      "Problem", "Problem definition class.",
      bp::init<const shared_ptr<Manifold> &, const CostPtr &,
               const std::vector<Constraint> &>(
          bp::args("self", "space", "cost", "constraints")))
      .def(bp::init<const shared_ptr<Manifold> &, const CostPtr &>(
          bp::args("self", "space", "cost")))
      .def_readwrite("cost", &Problem::cost_, "The cost function instance.")
      .def_readwrite("manifold", &Problem::manifold_, "Problem manifold.")
      .add_property("num_constraint_blocks", &Problem::getNumConstraints,
                    "Get the number of constraint blocks.")
      .add_property("total_constraint_dim", &Problem::getTotalConstraintDim,
                    "Get the total dimension of the constraints.")
      .add_property("nx", &Problem::nx, "Get the problem tangent space dim.")
      .add_property("ndx", &Problem::ndx, "Get the problem tangent space dim.")
      .def("add_constraint", &Problem::addConstraint<const Constraint &>,
           bp::args("self", "cstr"), "Add a constraint to the problem.");
}

} // namespace python
} // namespace proxnlp
