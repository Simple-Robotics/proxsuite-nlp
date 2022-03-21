#pragma once

#include "lienlp/fwd.hpp"

#include <eigenpy/eigen-typedef.hpp>


namespace lienlp {
namespace python {

  namespace context {
    
    using Scalar = double;
    enum { Options = 0 };

    EIGENPY_MAKE_TYPEDEFS_ALL_SIZES(Scalar, Options, s)

    using Problem_t = Problem<Scalar>;
    using Result_t = SResults<Scalar>;

  } // namespace context

} // namespace python
} // namespace lienlp


