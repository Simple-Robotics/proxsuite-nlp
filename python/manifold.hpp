#pragma once

#include <pinocchio/fwd.hpp>
#include "proxnlp/python/fwd.hpp"
#include "proxnlp/manifold-base.hpp"


namespace proxnlp
{
namespace python
{

  namespace internal
  {

    
    /// Expose the base manifold type
    void exposeBaseManifold()
    {
      using context::Scalar;
      using context::VectorXs;
      using context::VectorRef;
      using context::ConstVectorRef;
      using context::MatrixXs;
      using context::MatrixRef;
      using context::Manifold;

      using BinaryFunTypeRet = VectorXs(Manifold::*)(const ConstVectorRef&, const ConstVectorRef&) const;
      using BinaryFunType    = void(Manifold::*)(const ConstVectorRef&, const ConstVectorRef&, VectorRef) const;
      using JacobianFunType    = void(Manifold::*)(const ConstVectorRef&,const ConstVectorRef&, MatrixRef, int) const;

      bp::register_ptr_to_python<shared_ptr<Manifold>>();
      bp::register_ptr_to_python<shared_ptr<const Manifold>>();

      bp::class_<Manifold, boost::noncopyable>(
        "ManifoldAbstract", "Manifold abstract class.", bp::no_init
      )
        .add_property("nx", &Manifold::nx,  "Manifold representation dimension.")
        .add_property("ndx",&Manifold::ndx, "Tangent space dimension.")
        .def("neutral", &Manifold::neutral, "Get the neutral point from the manifold (if a Lie group).")
        .def("rand",    &Manifold::rand,    "Sample a random point from the manifold.")
        .def("integrate", static_cast<BinaryFunType>(&Manifold::integrate),    bp::args("self", "x", "v", "out"))
        .def("difference",static_cast<BinaryFunType>(&Manifold::difference),   bp::args("self", "x0", "x1", "out"))
        .def("integrate", static_cast<BinaryFunTypeRet>(&Manifold::integrate), bp::args("self", "x", "v"))
        .def("difference",static_cast<BinaryFunTypeRet>(&Manifold::difference),bp::args("self", "x0", "x1"))
        .def("interpolate",
             (void(Manifold::*)(const ConstVectorRef&,const ConstVectorRef&,const Scalar&, VectorRef) const)(&Manifold::interpolate),
             bp::args("self", "x0", "x1", "u", "out"))
        .def("interpolate", (VectorXs(Manifold::*)(const ConstVectorRef&,const ConstVectorRef&,const Scalar&) const)(&Manifold::interpolate),
             bp::args("self", "x0", "x1", "u"),
             "Interpolate between two points on the manifold. Allocated version.")
        .def("Jintegrate",  static_cast<JacobianFunType>(&Manifold::Jintegrate),
             bp::args("self", "x", "v", "Jout", "arg"),
             "Compute the Jacobian of the exp operator.")
        .def("Jdifference", static_cast<JacobianFunType>(&Manifold::Jdifference),
             bp::args("self", "x0", "x1", "Jout", "arg"),
             "Compute the Jacobian of the log operator.")
        .def("Jintegrate",
             +[](const Manifold& m, const ConstVectorRef x, const ConstVectorRef& v, int arg) {
               MatrixXs Jout(m.ndx(), m.ndx());
               m.Jintegrate(x, v, Jout, arg);
               return Jout; },
             "Compute and return the Jacobian of the exp.")
        .def("JintegrateTransport", &Manifold::JintegrateTransport,
             bp::args("self", "x", "v", "J", "arg"),
             "Perform parallel transport of matrix J expressed at point x+v to point x.")
        .def("Jdifference",
             +[](const Manifold& m, const ConstVectorRef x0, const ConstVectorRef& x1, int arg) {
               MatrixXs Jout(m.ndx(), m.ndx());
               m.Jdifference(x0, x1, Jout, arg);
               return Jout; },
             "Compute and return the Jacobian of the log.")
        .def("tangent_space", &Manifold::tangentSpace, bp::args("self"), "Returns an object representing the tangent space to this manifold.")
        .def(bp::self * bp::self)  // multiplication operator
        ;

    }
  } // namespace internal
  

} // namespace python

  
} // namespace proxnlp
