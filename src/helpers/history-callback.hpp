#pragma once

#include "lienlp/helpers-base.hpp"
#include "lienlp/solver-base.hpp"


namespace lienlp
{
  namespace helpers
  {
    /** @brief  Store the history of results.
     */
    template<typename Scalar>
    struct history_callback : callback<Scalar>
    {
      using Result_t = SResults<Scalar>;
      history_callback(const SWorkspace<Scalar>& workspace,
                       const Result_t& results,
                       bool store_pd_vars=true,
                       bool store_values=true,
                       bool store_residuals=true)
                       : workspace_(workspace)
                       , results_(results)
                       , store_primal_dual_vars_(store_pd_vars)
                       , store_values_(store_values)
                       , store_residuals_(store_residuals)
      {}

      struct
      {
        std::vector<typename Result_t::VectorXs> xs;
        std::vector<typename Result_t::VectorOfVectors> lams;
        std::vector<Scalar> values;
        std::vector<Scalar> prim_infeas;
        std::vector<Scalar> dual_infeas;
      } storage;

      void call()
      {
        if (store_primal_dual_vars_)
        {
          storage.xs.push_back(results_.xOpt);
          storage.lams.push_back(results_.lamsOpt);
        }
        if (store_values_)
          storage.values.push_back(results_.value);
        if (store_residuals_)
        {
          storage.prim_infeas.push_back(workspace_.primalInfeas);
          storage.dual_infeas.push_back(workspace_.dualInfeas);
        }
      }

    protected:
      const SWorkspace<Scalar>& workspace_;
      const Result_t& results_;
      const bool store_primal_dual_vars_;
      const bool store_values_;
      const bool store_residuals_;
    };
    
  } // namespace helpers
} // namespace lienlp

