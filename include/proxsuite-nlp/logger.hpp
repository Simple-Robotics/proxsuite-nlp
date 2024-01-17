/// @file
/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#pragma once

#include <vector>
#include <array>
#include <fmt/color.h>
#include <fmt/ranges.h>

namespace proxsuite {
namespace nlp {

constexpr int NUM_KEYS = 10;
const std::array<std::string, NUM_KEYS> BASIC_KEYS = {
    "iter", "step_size", "inner_crit", "prim_err", "dual_err",
    "xreg", "dphi0",     "merit",      "dM",       "al_iter"};
constexpr char int_format[] = "{: >{}d}";
constexpr char sci_format[] = "{: > {}.{}e}";
constexpr char dbl_format[] = "{: > {}.{}g}";

struct LogRecord {
  size_t iter;
  double step_size;
  double inner_crit;
  double prim_err;
  double dual_err;
  double xreg;
  double dphi0;
  double merit;
  double dM;
  size_t al_iter;
};

/// @brief  A logging utility.
struct BaseLogger {
  unsigned int COL_WIDTH_0 = 4;
  unsigned int COL_WIDTH = 10;
  bool active = true;

  constexpr static std::size_t print_outline_every() { return 25; }

  const std::string join_str = "｜";

  void start() const {
    if (!active)
      return;
    static constexpr char fstr[] = "{:^{}s}";
    std::array<std::string, NUM_KEYS> v;
    v[0] = fmt::format(fstr, BASIC_KEYS[0], COL_WIDTH_0);
    for (std::size_t i = 1; i < BASIC_KEYS.size(); ++i) {
      v[i] = fmt::format(fstr, BASIC_KEYS[i], COL_WIDTH);
    }
    fmt::print(fmt::emphasis::bold, "{}\n", fmt::join(v, join_str));
  }

  void log(const LogRecord &values) const {
    if (!active)
      return;
    std::vector<std::string> v;
    int sci_prec = 3;
    int dbl_prec = 3;
    using fmt::format;
    if (values.iter % print_outline_every() == 0)
      start();
    v.push_back(format(int_format, values.iter, COL_WIDTH_0));
    v.push_back(format(sci_format, values.step_size, COL_WIDTH, sci_prec));
    v.push_back(format(sci_format, values.inner_crit, COL_WIDTH, sci_prec));
    v.push_back(format(sci_format, values.prim_err, COL_WIDTH, sci_prec));
    v.push_back(format(sci_format, values.dual_err, COL_WIDTH, sci_prec));
    v.push_back(format(sci_format, values.xreg, COL_WIDTH, sci_prec));
    v.push_back(format(sci_format, values.dphi0, COL_WIDTH, dbl_prec));
    v.push_back(format(sci_format, values.merit, COL_WIDTH, sci_prec));
    v.push_back(format(dbl_format, values.dM, COL_WIDTH, dbl_prec));
    v.push_back(format(int_format, values.al_iter, COL_WIDTH));

    fmt::print("{}\n", fmt::join(v, join_str));
  }
};

} // namespace nlp
} // namespace proxsuite
