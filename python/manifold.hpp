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
      using context::VectorRef;
      using context::ConstVectorRef;
      using context::MatrixRef;
      using context::Manifold;
      using PointType = typename Manifold::PointType;
      using VecType = typename Manifold::TangentVectorType;

      using IntegrateRetType  = PointType(Manifold::*)(const ConstVectorRef&, const ConstVectorRef&) const;
      using IntegrateFun_t    = void     (Manifold::*)(const ConstVectorRef&, const ConstVectorRef&, VectorRef) const;
      using DifferenceRetType = VecType  (Manifold::*)(const ConstVectorRef&, const ConstVectorRef&) const;
      using DifferenceFun_t   = void     (Manifold::*)(const ConstVectorRef&, const ConstVectorRef&, VectorRef) const;

      bp::register_ptr_to_python<shared_ptr<Manifold>>();

      bp::class_<Manifold, shared_ptr<Manifold>, boost::noncopyable>(
        "ManifoldAbstract", "Manifold abstract class.",
        bp::no_init
      )
        .add_property("nx", &Manifold::nx, "Manifold representation dimension.")
        .add_property("ndx", &Manifold::ndx, "Tangent space dimension.")
        .def("neutral", &Manifold::neutral, "Get the neutral point from the manifold (if a Lie group).")
        .def("rand", &Manifold::rand, "Sample a random point from the manifold.")
        .def("integrate", static_cast<IntegrateRetType>  (&Manifold::integrate),  bp::args("x", "v"))
        .def("integrate", static_cast<IntegrateFun_t>    (&Manifold::integrate),  bp::args("x", "v", "out"))
        .def("difference", static_cast<DifferenceRetType>(&Manifold::difference), bp::args("x0", "x1"))
        .def("difference", static_cast<DifferenceFun_t>  (&Manifold::difference), bp::args("x0", "x1", "out"))
        .def("interpolate",
             (void(Manifold::*)(const ConstVectorRef&,const ConstVectorRef&,const Scalar&, VectorRef) const)(&Manifold::interpolate),
             bp::args("self", "x0", "x1", "u", "out"))
        .def("interpolate", (PointType(Manifold::*)(const ConstVectorRef&,const ConstVectorRef&,const Scalar&) const)(&Manifold::interpolate),
             bp::args("self", "x0", "x1", "u"),
             "Interpolate between two points on the manifold. Allocated version.")
        .def("Jintegrate",
             (void(Manifold::*)(const ConstVectorRef&, const ConstVectorRef&, MatrixRef, int) const)&Manifold::Jintegrate,
             bp::args("self", "x", "v", "Jout", "arg"),
             "Compute the Jacobian of the integration operator.")
        .def("Jdifference",
             (void(Manifold::*)(const ConstVectorRef&, const ConstVectorRef&, MatrixRef, int) const)&Manifold::Jdifference,
             bp::args("self", "x0", "x1", "Jout", "arg"),
             "Compute the Jacobian of the difference operator.")
        .def("tangent_space", &Manifold::tangentSpace, bp::args("self"), "Returns an object representing the tangent space to this manifold.")
        .def(bp::self * bp::self)  // multiplication operator
        ;

    }
  } // namespace internal
  

} // namespace python

  
} // namespace proxnlp
