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
      history_callback(bool store_pd_vars=true,
                       bool store_values=true,
                       bool store_residuals=true)
                       : store_primal_dual_vars_(store_pd_vars)
                       , store_values_(store_values)
                       , store_residuals_(store_residuals)
      {}

      struct
      {
        std::vector<typename SResults<Scalar>::VectorXs> xs;
        std::vector<typename SResults<Scalar>::VectorXs> lams_data;
        std::vector<typename SResults<Scalar>::VectorOfRef> lams;
        std::vector<Scalar> values;
        std::vector<Scalar> prim_infeas;
        std::vector<Scalar> dual_infeas;
      } storage;

      void call(const SWorkspace<Scalar>& workspace,
                const SResults<Scalar>& results)
      {
        if (store_primal_dual_vars_)
        {
          storage.xs.push_back(results.xOpt);
          storage.lams_data.push_back(results.lamsOpt_d);
          storage.lams.push_back(results.lamsOpt);
        }
        if (store_values_)
          storage.values.push_back(results.value);
        if (store_residuals_)
        {
          storage.prim_infeas.push_back(workspace.primalInfeas);
          storage.dual_infeas.push_back(workspace.dualInfeas);
        }
      }

    protected:
      const bool store_primal_dual_vars_;
      const bool store_values_;
      const bool store_residuals_;
    };
    
  } // namespace helpers
} // namespace lienlp

