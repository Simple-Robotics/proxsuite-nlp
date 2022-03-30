#pragma once

#include "lienlp/fwd.hpp"


namespace lienlp
{
  namespace helpers
  {
    
    template<typename Scalar>
    struct callback
    {
      virtual void call() = 0;
      virtual ~callback() = default;
    };

  } // namespace helpers
} // namespace lienlp

