#pragma once

#include "lienlp/fwd.hpp"


namespace lienlp
{
  /// Helper functions and structs.
  namespace helpers
  {
    
    template<typename Scalar>
    struct callback
    {
      virtual void call(const SWorkspace<Scalar>&, const SResults<Scalar>&) = 0;
      virtual ~callback() = default;
    };

  } // namespace helpers
} // namespace lienlp

