#pragma once

#include "proxnlp/manifold-base.hpp"

#include <type_traits>


namespace proxnlp
{
  template<typename _Scalar, int _Dim, int _Options>
  struct VectorSpaceTpl : public ManifoldAbstractTpl<_Scalar, _Options>
  {
    using Scalar = _Scalar;
    enum
    {
      Dim = _Dim,
      Options = _Options
    };
    using Base = ManifoldAbstractTpl<Scalar, Options>;
    PROXNLP_DEFINE_MANIFOLD_TYPES(Base)

    const int dim_;

    /// @brief    Default constructor where the dimension is supplied.
    template<int N = Dim,
             typename = typename std::enable_if<N == Eigen::Dynamic>::type>
    VectorSpaceTpl(const int dim) : dim_(dim) {}

    /// @brief    Default constructor without arguments.
    ///
    /// @details  This constructor is disabled for the dynamic-sized vectors.
    template<int N = Dim,
             typename = typename std::enable_if<N != Eigen::Dynamic>::type>
    VectorSpaceTpl() : dim_(Dim) {}

    inline int nx()  const { return dim_; }
    inline int ndx() const { return dim_; }

    /// \name implementations

    /* Integrate */

    void integrate_impl(const ConstVectorRef& x,
                        const ConstVectorRef& v,
                        VectorRef out) const
    {
      out = x + v;
    }

    void Jintegrate_impl(const ConstVectorRef&,
                         const ConstVectorRef&,
                         MatrixRef Jout,
                         int) const
    {
      Jout.setIdentity();
    }

    /* Difference */

    void difference_impl(const ConstVectorRef& x0,
                         const ConstVectorRef& x1,
                         VectorRef out) const
    {
      out = x1 - x0;
    }

    void Jdifference_impl(const ConstVectorRef&,
                          const ConstVectorRef&,
                          MatrixRef Jout,
                          int arg) const
    {
      switch (arg)
      {
      case 0:   Jout = -MatrixXs::Identity(ndx(), ndx()); break;
      case 1:   Jout.setIdentity(); break;
      default:  throw std::runtime_error("Wrong arg value.");
      }
    }

    void interpolate_impl(const ConstVectorRef& x0,
                          const ConstVectorRef& x1,
                          const Scalar& u,
                          VectorRef out) const
    {
      out = u * x1 + (static_cast<Scalar>(1.) - u) * x0;
    }

  };
  
} // namespace proxnlp
