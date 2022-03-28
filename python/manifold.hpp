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
      using Man = context::ManifoldAbstract_t;
      using PointType = typename Man::PointType;
      using VecType = typename Man::TangentVectorType;

      using IntegrateRet_t = PointType(Man::*)(const ConstVectorRef&, const ConstVectorRef&) const;
      using IntegrateFun_t = void     (Man::*)(const ConstVectorRef&, const ConstVectorRef&, context::VectorRef) const;
      using DifferenceRet_t = VecType(Man::*)(const ConstVectorRef&, const ConstVectorRef&) const;
      using DifferenceFun_t = void   (Man::*)(const ConstVectorRef&, const ConstVectorRef&, context::VectorRef) const;

      bp::class_<Man, shared_ptr<Man>, boost::noncopyable>(
        "ManifoldAbstract", "Manifold abstract class.",
        bp::no_init
      )
        .add_property("nx", &Man::nx, "Manifold representation dimension.")
        .add_property("ndx", &Man::ndx, "Tangent space dimension.")
        .def("neutral", &Man::neutral, "Get the neutral point from the manifold (if a Lie group).")
        .def("rand", &Man::rand, "Sample a random point from the manifold.")
        .def("integrate", static_cast<IntegrateRet_t>  (&Man::integrate), bp::args("x", "v"))
        .def("integrate", static_cast<IntegrateFun_t>  (&Man::integrate), bp::args("x", "v", "out"))
        .def("difference", static_cast<DifferenceRet_t>(&Man::difference), bp::args("x0", "x1"))
        .def("difference", static_cast<DifferenceFun_t>(&Man::difference), bp::args("x0", "x1", "out"))
        ;

    }
  } // namespace internal
  

} // namespace python

  
} // namespace lienlp
