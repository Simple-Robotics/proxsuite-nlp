#include "lienlp/python/fwd.hpp"
#include "lienlp/manifold-base.hpp"

#include "lienlp/modelling/spaces/pinocchio-groups.hpp"


namespace lienlp
{
namespace python
{


  template<typename LieGroup>
  void exposeLieGroup(const char* name, const char* docstring)
  {
    using context::VectorXs;
    using Out_t = PinocchioLieGroup<LieGroup>;
    using PointType = typename Out_t::PointType;
    using VecType = typename Out_t::TangentVectorType;
    using Arity2VecFunRet_t = PointType(const VectorXs&,
                                        const VectorXs&) const;

    // Arity2VecFunRet_t Out_t::*int1 = &Out_t::template integrate<VectorXs, VectorXs>;
    // VecType (Out_t::*diff1)(const context::VectorXs&, const context::VectorXs&) const = &Out_t::difference;

    bp::class_<Out_t>(
      name, docstring,
      bp::init<>())  // default ctor
      .add_property("nx", &Out_t::nx, "Manifold representation dimension.")
      .add_property("ndx", &Out_t::ndx, "Tangent space dimension.")
      .def("neutral", &Out_t::neutral)
      .def("rand", &Out_t::rand)
      // .def("integrate", int1)
      // .def<Arity2VecFunRet_t>("difference", &Out_t::difference)
      ;

  }

  void exposeManifold()
  {
    namespace pin = pinocchio;

    using VectorSpace = pin::VectorSpaceOperationTpl<Eigen::Dynamic, context::Scalar>;

    exposeLieGroup<pin::SpecialOrthogonalOperationTpl<2,context::Scalar>>("SO2", "SO(2) special orthogonal group.");
    exposeLieGroup<pin::SpecialOrthogonalOperationTpl<3,context::Scalar>>("SO3", "SO(3) special orthogonal group.");

  }

} // namespace python
} // namespace lienlp
