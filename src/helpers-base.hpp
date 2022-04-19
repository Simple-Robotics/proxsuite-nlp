#pragma once

#include "lienlp/fwd.hpp"


namespace lienlp
{
  /// Helper functions and structs.
  namespace helpers
  {
    
    template<typename Scalar>
    struct base_callback
    {
      virtual void call(const WorkspaceTpl<Scalar>&, const ResultsTpl<Scalar>&) = 0;
      virtual ~base_callback() = default;
    };

  } // namespace helpers
} // namespace lienlp

