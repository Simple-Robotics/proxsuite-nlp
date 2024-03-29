#ifdef PROXSUITE_NLP_WITH_PINOCCHIO
#include "proxsuite-nlp/python/residuals.hpp"

#include "proxsuite-nlp/modelling/residuals/rigid-transform-point.hpp"

namespace proxsuite {
namespace nlp {
namespace python {

using context::Scalar;

// fwd declaration
void exposePinocchioResiduals() {
  using RigidTransformPointAction = RigidTransformationPointActionTpl<Scalar>;
  expose_function<RigidTransformPointAction>(
      "RigidTransformationPointAction",
      "A residual representing the action :math:`M\\cdot p = Rp + t` of a "
      "rigid "
      "transform :math:`M` on a 3D point :math:`p`.",
      bp::init<context::Vector3s>(bp::args("self", "point")))
      .def_readonly("space", &RigidTransformPointAction::space_,
                    "Function input space.")
      .def_readwrite("point", &RigidTransformPointAction::point_)
      .add_property("skew_matrix", &RigidTransformPointAction::skew_point);
}

} // namespace python
} // namespace nlp
} // namespace proxsuite
#endif
