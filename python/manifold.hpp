#pragma once

#include <pinocchio/fwd.hpp>
#include "lienlp/python/fwd.hpp"
#include "lienlp/manifold-base.hpp"

// manifold visitor will also expose manifold-templated functors
#include "lienlp/python/manifold-functors.hpp"


namespace lienlp
{
namespace python
{
  
  template<typename Man>
  struct manifold_visitor : public bp::def_visitor<manifold_visitor<Man>>
  {
    using VectorXs = context::VectorXs;
    using PointType = typename Man::PointType;
    using VecType = typename Man::TangentVectorType;
    using IntegrateFun_t = PointType(const Eigen::MatrixBase<VectorXs>&, const Eigen::MatrixBase<VectorXs>&) const;
    using DifferenceFun_t = VecType(const Eigen::MatrixBase<VectorXs>&, const Eigen::MatrixBase<VectorXs>&) const;


    template<class PyClass>
    void visit(PyClass& cl) const
    {
      IntegrateFun_t Man::*int1 = &Man::template integrate<VectorXs, VectorXs>;
      DifferenceFun_t Man::*diff1 = &Man::template difference<VectorXs, VectorXs>;

      cl
        .add_property("nx", &Man::nx, "Manifold representation dimension.")
        .add_property("ndx", &Man::ndx, "Tangent space dimension.")
        .def("neutral", &Man::neutral, "Get the neutral point from the manifold (if a Lie group).")
        .def("rand", &Man::rand, "Sample a random point from the manifold.")
        .def("integrate", int1, bp::args("x", "v"))
        .def("difference", diff1, bp::args("x0", "x1"))
        ;

      ResidualVisitor<Man>::expose();

    }
  };

} // namespace python

  
} // namespace lienlp
