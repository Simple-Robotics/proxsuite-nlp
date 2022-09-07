/// @file linesearch-base.hpp
/// @brief  Implements a basic Armijo back-tracking strategy.
#pragma once

#include "proxnlp/linesearch-base.hpp"
#include "proxnlp/exceptions.hpp"

namespace proxnlp {

/// @brief Polynomials represented by their coefficients in decreasing order of
/// degree.
template <typename T> struct PolynomialTpl {
  using VectorXs = typename math_types<T>::VectorXs;
  VectorXs coeffs;
  PolynomialTpl() {}
  PolynomialTpl(const VectorXs &c) : coeffs(c) {}
  /// @brief Polynomial degree (number of coefficients minus one).
  std::size_t degree() const { return coeffs.size() - 1; }
  inline T evaluate(T a) const {
    T r = 0.0;
    for (int i = 0; i < coeffs.size(); i++) {
      r = r * a + coeffs(i);
    }
    return r;
  }
  PolynomialTpl derivative() const {
    if (degree() == 0) {
      return PolynomialTpl({0.});
    }
    VectorXs out(degree());
    for (int i = 0; i < coeffs.size() - 1; i++) {
      out(i) = coeffs(i) * (degree() - i);
    }
    return PolynomialTpl(out);
  }
};

/// @brief  Basic backtracking Armijo line-search strategy.
template <typename Scalar> class ArmijoLinesearch : public Linesearch<Scalar> {
public:
  using Base = Linesearch<Scalar>;
  using FunctionSample = typename Base::FunctionSample;
  using Polynomial = PolynomialTpl<Scalar>;
  using VectorXs = typename math_types<Scalar>::VectorXs;
  using Base::options;

  ArmijoLinesearch(const typename Base::Options &options) : Base(options) {}

  template <typename Fn>
  Scalar run(Fn phi, const Scalar phi0, const Scalar dphi0,
             Scalar &alpha_try) const {
    auto eval_sample_nograd = [&](const Scalar a) {
      return FunctionSample{a, phi(a)};
    };

    const FunctionSample lowerbound(0., phi0, dphi0);

    alpha_try = 1.;
    FunctionSample latest = eval_sample_nograd(alpha_try);
    FunctionSample previous;

    if (std::abs(dphi0) < options().dphi_thresh) {
      return phi0;
    }

    for (std::size_t i = 0; i < options().max_num_steps; i++) {

      const Scalar dM = latest.phi - phi0;
      if (dM <= options().armijo_c1 * alpha_try * dphi0) {
        break;
      }

      LSInterpolation strat = options().interp_type;
      if (strat == LSInterpolation::BISECTION) {
        alpha_try *= 0.5;
      } else {
        std::vector<FunctionSample> samples;
        samples.push_back(lowerbound);

        // interpolation routines
        switch (strat) {
        case LSInterpolation::QUADRATIC: {
          // two point interpolation: value, derivative at 0
          // and latest value
          samples.push_back(latest);
          break;
        }
        case LSInterpolation::CUBIC: {
          // three point interpolation: phi(0), phi'(0)
          // and last two values
          samples.push_back(latest);
          if (previous.valid) {
            samples.push_back(previous);
          }
          break;
        }
        default:
          proxnlp_runtime_error(
              "Unrecognized interpolation type in this context.\n");
          break;
        }

        alpha_try = this->minimize_interpolant(
            strat, samples, options().contraction_min * alpha_try,
            options().contraction_max * alpha_try);
      }
      alpha_try = std::max(alpha_try, options().alpha_min);
      previous = latest;
      latest = eval_sample_nograd(alpha_try);
      if (alpha_try <= options().alpha_min) {
        break;
      }
    }
    return latest.phi;
  }

  /// Propose a new candidate step size through safeguarded interpolation
  Scalar minimize_interpolant(LSInterpolation strat,
                              const std::vector<FunctionSample> &samples,
                              Scalar min_step_size,
                              Scalar max_step_size) const {
    Scalar anext = 0.0;
    Polynomial interpol;

    assert(samples.size() >= 2);
    const FunctionSample &lowerbound = samples[0];
    const Scalar &phi0 = lowerbound.phi;
    const Scalar &dphi0 = lowerbound.dphi;

    if (samples.size() == 2) {
      strat = LSInterpolation::QUADRATIC;
    }

    switch (strat) {
    case LSInterpolation::QUADRATIC: {

      const FunctionSample &cand0 = samples[1];
      Scalar a = (cand0.phi - phi0 - cand0.alpha * dphi0) /
                 (cand0.alpha * cand0.alpha);
      VectorXs coeffs(3);
      coeffs << a, dphi0, phi0;
      interpol = Polynomial(coeffs);
      assert(interpol.degree() == 2);
      anext = -dphi0 / (2. * a);
      break;
    }
    case LSInterpolation::CUBIC: {
      using Matrix2s = Eigen::Matrix<Scalar, 2, 2>;
      using Vector2s = Eigen::Matrix<Scalar, 2, 1>;
      Matrix2s alph_mat;
      Vector2s alph_rhs;
      Vector2s coeffs_cubic_interpolant;

      const FunctionSample &cand0 = samples[1];
      const FunctionSample &cand1 = samples[2];
      const Scalar &a0 = cand0.alpha;
      const Scalar &a1 = cand1.alpha;
      alph_mat(0, 0) = a0 * a0;
      alph_mat(0, 1) = -a1 * a1;
      alph_mat(1, 0) = -a0 * a0 * a0;
      alph_mat(1, 1) = a1 * a1 * a1;

      const Scalar mat_denom = a0 * a0 * a1 * a1 * (a1 - a0);
      alph_mat /= mat_denom;

      alph_rhs(0) = cand1.phi - phi0 - dphi0 * a1;
      alph_rhs(1) = cand0.phi - phi0 - dphi0 * a0;
      coeffs_cubic_interpolant = alph_mat * alph_rhs;

      const Scalar c3 = coeffs_cubic_interpolant(0);
      const Scalar c2 = coeffs_cubic_interpolant(1);
      VectorXs coeffs(4);
      coeffs << c3, c2, dphi0, phi0;
      interpol = Polynomial(coeffs);
      assert(interpol.degree() == 3);

      // minimizer of cubic interpolant -> solve dinterp/da = 0
      anext = (-c2 + std::sqrt(c2 * c2 - 3.0 * c3 * dphi0)) / (3.0 * c3);
      break;
    }
    default:
      break;
    }

    if ((anext > max_step_size) || (anext < min_step_size)) {
      // if min outside of (amin; amax), look at the edges
      Scalar pleft = interpol.evaluate(min_step_size);
      Scalar pright = interpol.evaluate(max_step_size);
      if (pleft <= pright) {
        anext = min_step_size;
      } else {
        anext = max_step_size;
      }
    }

    return anext;
  }
};

} // namespace proxnlp
