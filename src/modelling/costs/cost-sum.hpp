#pragma once


#include "lienlp/macros.hpp"
#include "lienlp/cost-function.hpp"


namespace lienlp {

  template<typename _Scalar>
  class CostSum : CostFunctionBase<_Scalar>
  {
  public:
    using Scalar = _Scalar;
    LIENLP_DEFINE_DYNAMIC_TYPES(Scalar)
    using Base = CostFunctionBase<Scalar>;
    using BasePtr = shared_ptr<Base>;

    std::vector<shared_ptr<Base>> m_components; /// component sub-costs
    std::vector<const Scalar> m_weights; /// cost component weights

    CostSum(const int nx, const int ndx,
            const std::vector<Scalar>& weights)
            : Base(nx, ndx), m_weights(weights)
    {}

    /// Constructor with a predefined vector of components.
    CostSum(const int nx, const int ndx,
            const std::vector<BasePtr>& comps,
            const std::vector<Scalar>& weights)
            : Base(nx, ndx), m_weights(weights), m_components(comps)
    {}

    void addComponent(BasePtr comp, const Scalar& w)
    {
      m_components.push_back(comp);
      m_weights.push_back(w);
      
    }

    Scalar operator()(const ConstVectorRef& x) const
    {
      Scalar result_ = 0.;
      for (std::size_t i = 0; i < m_components.size(); i++)
      {
        result_ += m_weights[i] * m_components[i]->operator()(x);
      }
      return result_;
    }

    void computeGradient(const ConstVectorRef& x, RefVector out) const
    {
      out.setZero();
      for (std::size_t i = 0; i < m_components.size(); i++)
      {
        out.noalias() = out + m_weights[i] * m_components[i]->computeGradient(x);
      }
    }

    void computeHessian(const ConstVectorRef& x, RefMatrix out) const
    {
      out.setZero();
      for (std::size_t i = 0; i < m_components.size(); i++)
      {
        out.noalias() = out + m_weights[i] * m_components[i]->computeHessian(x);
      }
    }

  };

} // namespace lienlp

