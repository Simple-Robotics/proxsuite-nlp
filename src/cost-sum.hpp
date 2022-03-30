#pragma once


#include "lienlp/macros.hpp"
#include "lienlp/cost-function.hpp"


namespace lienlp
{

  template<typename _Scalar>
  struct CostSum : CostFunctionBase<_Scalar>
  {
  public:
    using Scalar = _Scalar;
    LIENLP_DYNAMIC_TYPEDEFS(Scalar)
    using Base = CostFunctionBase<Scalar>;
    using BaseRef = std::reference_wrapper<Base>;

    std::vector<BaseRef> m_components; /// component sub-costs
    std::vector<Scalar> m_weights; /// cost component weights

    CostSum(int nx, int ndx) : Base(nx, ndx) {}

    /// Constructor with a predefined vector of components.
    CostSum(int nx, int ndx,
            const std::vector<BaseRef>& comps,
            const std::vector<Scalar>& weights)
            : Base(nx, ndx)
            , m_components(comps)
            , m_weights(weights)
    {}

    std::size_t numComponents() const
    {
      return m_components.size();
    }

    Scalar call(const ConstVectorRef& x) const
    {
      Scalar result_ = 0.;
      for (std::size_t i = 0; i < numComponents(); i++)
      {
        result_ += m_weights[i] * m_components[i].get().call(x);
      }
      return result_;
    }

    void computeGradient(const ConstVectorRef& x, VectorRef out) const
    {
      out.setZero();
      for (std::size_t i = 0; i < numComponents(); i++)
      {
        out.noalias() = out + m_weights[i] * m_components[i].get().computeGradient(x);
      }
    }

    void computeHessian(const ConstVectorRef& x, MatrixRef out) const
    {
      out.setZero();
      for (std::size_t i = 0; i < numComponents(); i++)
      {
        out.noalias() = out + m_weights[i] * m_components[i].get().computeHessian(x);
      }
    }

    void addComponent(Base& comp, const Scalar& w)
    {
      m_components.push_back(std::ref(comp));
      m_weights.push_back(w);
    }

    void addComponent(Base& comp)
    {
      addComponent(comp, 1.);
    }

    void operator+=(Base& other)
    {
      addComponent(other);
    }

    void operator+=(CostSum<Scalar>& other)
    {
      m_components.insert(m_components.end(), other.m_components.begin(), other.m_components.end());
      m_weights.insert(m_weights.end(), other.m_weights.begin(), other.m_weights.end());
    }

  };

  template<typename Scalar>
  CostSum<Scalar> operator+(CostFunctionBase<Scalar>& left, CostFunctionBase<Scalar>& right)
  {
    assert((left.nx() == right.nx()) && (left.ndx() == right.ndx()) &&
           "Left and right should have the same input spaces.");
    CostSum<Scalar> out(left.nx(), left.ndx());
    out += left;
    out += right;
    return out;
  }

} // namespace lienlp

