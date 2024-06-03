/// @file
/// @copyright Copyright (C) 2024 INRIA
#pragma once

#include <fmt/ostream.h>
#include <fmt/ranges.h>

namespace fmt {
template <> struct formatter<proxsuite::nlp::HessianApprox> {
  template <typename ParseContext> constexpr auto parse(ParseContext &ctx) {
    return ctx.begin();
  }

  template <typename FormatContext>
  auto format(const proxsuite::nlp::HessianApprox &hessian_approx,
              FormatContext &ctx) const {
    std::string name;
    switch (hessian_approx) {
    case proxsuite::nlp::HessianApprox::EXACT:
      name = "EXACT";
      break;
    case proxsuite::nlp::HessianApprox::GAUSS_NEWTON:
      name = "GAUSS_NEWTON";
      break;
    case proxsuite::nlp::HessianApprox::BFGS:
      name = "BFGS";
      break;
    case proxsuite::nlp::HessianApprox::IDENTITY:
      name = "IDENTITY";
      break;
    }
    return format_to(ctx.out(), "{}", name);
  }
};
} // namespace fmt
