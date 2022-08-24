/// @file linesearch-base.hpp
/// @brief  Base structs for linesearch algorithms.
#pragma once

#include "proxnlp/workspace.hpp"
#include "proxnlp/results.hpp"

#include <fmt/core.h>
#include <ostream>

namespace proxnlp {
enum LinesearchStrategy { ARMIJO, CUBIC_INTERP };
enum class LSInterpolation { BISECTION, QUADRATIC, CUBIC };

/// @brief Base linesearch class.
/// Design pattern inspired by Google Ceres-Solver.
template <typename T> class Linesearch {
public:
  struct Options {
    Options()
        : armijo_c1(1e-4), wolfe_c2(0.9), dphi_thresh(1e-13), alpha_min(1e-6),
          max_num_steps(20), verbosity(VerboseLevel::QUIET),
          contraction_min(0.5), contraction_max(0.8) {}
    T armijo_c1;
    T wolfe_c2;
    T dphi_thresh;
    T alpha_min;
    std::size_t max_num_steps;
    VerboseLevel verbosity;
    LSInterpolation interp_type = LSInterpolation::BISECTION;
    T contraction_min;
    T contraction_max;
  };
  explicit Linesearch(const Linesearch::Options &options);
  ~Linesearch();

  struct FunctionSample {
    T alpha;
    T phi;
    T dphi;
    bool valid;
    FunctionSample() : alpha(0.), phi(0.), dphi(0.), valid(false) {}
    FunctionSample(T a, T v) : alpha(a), phi(v), dphi(0.), valid(true) {}
    FunctionSample(T a, T v, T g) : alpha(a), phi(v), dphi(g), valid(true) {}
  };

  const Linesearch::Options &options() const { return options_; }

private:
  Linesearch::Options options_;
};

} // namespace proxnlp

#include "proxnlp/linesearch-armijo.hpp"
// #include "proxnlp/linesearch-cubic-interp.hpp"

namespace proxnlp {

template <typename T>
Linesearch<T>::Linesearch(const Linesearch::Options &options)
    : options_(options) {}

template <typename T> Linesearch<T>::~Linesearch() = default;

} // namespace proxnlp
