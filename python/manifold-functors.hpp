/**
 * @file    Utility classes to expose manifold-templated functors.
 */
#pragma once

#include "lienlp/python/fwd.hpp"
#include "lienlp/modelling/residuals/state-residual.hpp"

namespace lienlp
{
namespace python
{
  namespace bp = boost::python;


  /// Expose manifold-templated functors and residuals.
  template<typename Man>
  struct ResidualVisitor
  {
    using VectorXs = context::VectorXs;
    using DFunctor_t = context::DFunctor_t;

    static void expose()
    {
      bp::class_<StateResidual<Man>, bp::bases<DFunctor_t>>(
        "StateResidual", "Residual distance to a fixed point on the manifold",
        bp::no_init
      )
        .def(bp::init<Man&, const context::ConstVectorRef&>(bp::args("space", "target")))
        ;
    }


  };
  
} // namespace python

  
} // namespace lienlp

