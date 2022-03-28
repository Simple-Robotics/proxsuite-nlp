#pragma once

#include "lienlp/manifold-base.hpp"
#include "lienlp/cost-function.hpp"
#include "lienlp/constraint-base.hpp"
#include "lienlp/modelling/constraints/equality-constraint.hpp"

#include <vector>


namespace lienlp
{

  template<typename _Scalar>
  struct Problem
  {
  public:
    using Scalar = _Scalar;
    LIENLP_DEFINE_DYNAMIC_TYPES(Scalar)

    /// Generic constraint type
    using ConstraintType = ConstraintSetBase<Scalar>;
    using ConstraintPtr = shared_ptr<ConstraintType>;
    /// Equality constraint type
    using EqualityType = EqualityConstraint<Scalar>;
    /// Cost function type
    using CostType = CostFunctionBase<Scalar>;

    /// The cost functional
    const CostType& m_cost;
    /// Get a pointer to the \f$i\f$-th constraint pointer
    const ConstraintPtr getConstraint(const std::size_t& i) const
    {
      return m_cstrs[i];
    }

    /// @brief Get the number of constraint blocks.
    std::size_t getNumConstraints() const
    {
      return m_cstrs.size();
    }

    int getTotalConstraintDim() const
    {
      return m_ncTotal;
    }

    std::vector<int> getConstraintDims() const
    {
      return m_ncs;
    }

    Problem(const CostType& cost) : m_cost(cost), m_ncTotal(0) {}

    Problem(const CostType& cost, const std::vector<ConstraintPtr>& constraints)
            : m_cost(cost), m_cstrs(constraints), m_ncTotal(0)
    {
      int& nc = const_cast<int&>(m_ncTotal);
      for (ConstraintPtr cstr : m_cstrs)
      {
        nc += cstr->nr();
        m_ncs.push_back(cstr->nr());
      }
    }
    
    /// @brief   Allocate a set of multipliers (or residuals) for a given problem instance.
    static void allocateMultipliersOrResiduals(
      const Problem<Scalar>& prob,
      VectorOfVectors& out)
    {
      out.reserve(prob.getNumConstraints());
      for (std::size_t i = 0; i < prob.getNumConstraints(); i++)
      {
        ConstraintPtr cur_cstr = prob.getConstraint(i);
        out.push_back(VectorXs::Zero(cur_cstr->nr()));
      }
    }
  protected:
    /// Vector of equality constraints.
    const std::vector<ConstraintPtr> m_cstrs;
    /// Total number of constraints
    const int m_ncTotal;
    std::vector<int> m_ncs;
  };

} // namespace lienlp


