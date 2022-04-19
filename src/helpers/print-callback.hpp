#pragma once

#include "lienlp/helpers-base.hpp"

namespace lienlp
{
  namespace helpers
  {

    template<typename Scalar>
    struct print_callback : base_callback<Scalar>
    {
      void call(const WorkspaceTpl<Scalar>& workspace,
                const ResultsTpl<Scalar>& results)
      {

      }

    };
    
  } // namespace helpers
} // namespace lienlp

