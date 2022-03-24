#pragma once

#include "lienlp/manifold-base.hpp"
#include "lienlp/cost-function.hpp"
#include "lienlp/constraint-base.hpp"
#include "lienlp/modelling/constraints/equality-constraint.hpp"

#include <vector>


namespace lienlp {

  template<typename _Scalar>
  struct Problem
  {
  public:
    using Scalar = _Scalar;
    LIENLP_DEFINE_DYNAMIC_TYPES(Scalar)

    /// Generic constraint type
    using CstrType = ConstraintSetBase<Scalar>;
    using CstrPtr = shared_ptr<CstrType>;
    /// Equality constraint type
    using EqualityType = EqualityConstraint<Scalar>;
    /// Cost function type
    using CostType = CostFunctionBase<Scalar>;

    /// The cost functional
    const CostType& m_cost;

    /// Get a pointer to the \f$i\f$-th constraint pointer
    const CstrPtr getConstraint(const std::size_t& i) const
    {
      return m_cstrs[i];
    }

    /// @brief Get the number of constraint blocks.
    std::size_t getNumConstraints() const
    {
      return m_cstrs.size();
    }

    int getNcTotal() const
    {
      return m_ncTotal;
    }

    Problem(const CostType& cost) : m_cost(cost) {}

    Problem(const CostType& cost,
            const std::vector<CstrPtr>& constraints)
            : m_cost(cost), m_cstrs(constraints)
    {
      m_ncTotal = 0;
      for (CstrPtr cstr : m_cstrs)
      {
        m_ncTotal += cstr->nr();
      }
    }


    static void allocateMultipliers(
      const Problem<Scalar>& prob,
      VectorOfVectors& out)
    {
      out.reserve(prob.getNumConstraints());
      for (std::size_t i = 0; i < prob.getNumConstraints(); i++)
      {
        CstrPtr cur_cstr = prob.getConstraint(i);
        out.push_back(VectorXs::Zero(cur_cstr->nr()));
      }
    }

  protected:
    /// Vector of equality constraints.
    const std::vector<CstrPtr> m_cstrs;
    /// Total number of constraints
    int m_ncTotal;
  };

} // namespace lienlp


