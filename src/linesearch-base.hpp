/// @file linesearch-base.hpp
/// @brief  Base structs for linesearch algorithms.
#pragma once

#include "proxnlp/workspace.hpp"
#include "proxnlp/results.hpp"

#include <fmt/core.h>

namespace proxnlp {
enum LinesearchStrategy { ARMIJO, CUBIC_INTERP };

/// @brief Base linesearch class.
/// Design pattern inspired by Google Ceres-Solver.
template <typename T> class Linesearch {
public:
  struct Options {
    Options()
        : armijo_c1(1e-4), wolfe_c2(0.9), dphi_thresh(1e-13),
          max_num_steps(20) {}
    T armijo_c1;
    T wolfe_c2;
    T dphi_thresh;
    std::size_t max_num_steps;
    VerboseLevel verbosity = VerboseLevel::QUIET;
  };
  ~Linesearch() = default;

  struct FunctionSample {
    T alpha;
    T phi;
    T dphi;
  };

  static Linesearch *create(LinesearchStrategy ls_strat);

  const Linesearch::Options &options() const { return options_; }

private:
  Linesearch::Options options_;
};

} // namespace proxnlp

#include "proxnlp/linesearch-armijo.hpp"
#include "proxnlp/linesearch-cubic-interp.hpp"

namespace proxnlp {


template<typename T>
Linesearch<T> *Linesearch<T>::create(LinesearchStrategy ls_strat) {
  Linesearch* out = nullptr;
  switch (ls_strat) {
    case ARMIJO: out = new ArmijoLinesearch<T>();
    case CUBIC_INTERP: out = new CubicInterpLinesearch<T>();
    default: break;
  }
  return out;
}
} // namespace proxnlp

