#include "proxsuite-nlp/python/fwd.hpp"
#include "proxsuite-nlp/problem-base.hpp"

namespace proxsuite {
namespace nlp {
namespace python {

void exposeProblem() {
  using context::Constraint;
  using context::Manifold;
  using context::Problem;

  bp::class_<Problem, shared_ptr<Problem>>(
      "Problem", "Problem definition class.", bp::no_init)
      .def(bp::init<shared_ptr<Manifold>, shared_ptr<context::Cost>,
                    const std::vector<Constraint> &>(
          (bp::arg("self"), bp::arg("space"), bp::arg("cost"),
           bp::arg("constraints") = bp::list())))
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
} // namespace nlp
} // namespace proxsuite
