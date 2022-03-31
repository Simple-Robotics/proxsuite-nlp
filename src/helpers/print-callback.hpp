#pragma once

#include "lienlp/helpers-base.hpp"

namespace lienlp
{
  namespace helpers
  {

    template<typename Scalar>
    struct print_callback : callback<Scalar>
    {
      void call(const SWorkspace<Scalar>& workspace,
                const SResults<Scalar>& results)
      {

      }

    };
    
  } // namespace helpers
} // namespace lienlp

