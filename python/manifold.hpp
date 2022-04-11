#pragma once

#include <pinocchio/fwd.hpp>
#include "lienlp/python/fwd.hpp"
#include "lienlp/manifold-base.hpp"


namespace lienlp
{
namespace python
{

  namespace internal
  {
    
    /// Expose the base manifold type
    void exposeBaseManifold()
    {
      using ConstVectorRef = context::ConstVectorRef;
      using Manifold = context::Manifold;
      using PointType = typename Manifold::PointType;
      using VecType = typename Manifold::TangentVectorType;

      using IntegrateRetType = PointType(Manifold::*)(const ConstVectorRef&, const ConstVectorRef&) const;
      using IntegrateFun_t = void     (Manifold::*)(const ConstVectorRef&, const ConstVectorRef&, context::VectorRef) const;
      using DifferenceRetType = VecType (Manifold::*)(const ConstVectorRef&, const ConstVectorRef&) const;
      using DifferenceFun_t = void    (Manifold::*)(const ConstVectorRef&, const ConstVectorRef&, context::VectorRef) const;

      bp::class_<Manifold, shared_ptr<Manifold>, boost::noncopyable>(
        "ManifoldAbstract", "Manifold abstract class.",
        bp::no_init
      )
        .add_property("nx", &Manifold::nx, "Manifold representation dimension.")
        .add_property("ndx", &Manifold::ndx, "Tangent space dimension.")
        .def("neutral", &Manifold::neutral, "Get the neutral point from the manifold (if a Lie group).")
        .def("rand", &Manifold::rand, "Sample a random point from the manifold.")
        .def("integrate", static_cast<IntegrateRetType>  (&Manifold::integrate), bp::args("x", "v"))
        .def("integrate", static_cast<IntegrateFun_t>    (&Manifold::integrate), bp::args("x", "v", "out"))
        .def("difference", static_cast<DifferenceRetType>(&Manifold::difference),bp::args("x0", "x1"))
        .def("difference", static_cast<DifferenceFun_t>  (&Manifold::difference),bp::args("x0", "x1", "out"))
        ;

    }
  } // namespace internal
  

} // namespace python

  
} // namespace lienlp
