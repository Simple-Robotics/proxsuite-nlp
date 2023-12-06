#include <iostream>
#include "Eigen/Cholesky"
#include <chrono>
#include "proxnlp/linalg/bunchkaufman.hpp"

template <typename F> auto time1(F f) -> double {
  auto start = std::chrono::steady_clock::now();
  f();
  auto end = std::chrono::steady_clock::now();
  return std::chrono::duration<double>(end - start).count();
}

template <typename F> auto timeit(F f) -> double {
  auto time_limit = 1e-0;
  unsigned n_iters = 1;
  while (true) {
    auto t = time1([&] {
      for (unsigned i = 0; i < n_iters; ++i) {
        f();
      }
    });

    if (t >= time_limit || n_iters > 1'000'000'000) {
      return t / n_iters;
    }

    n_iters = 2 * std::max<unsigned>(n_iters, time_limit / t);
  }
}

auto main() -> int {
  std::srand(0);
  using Eigen::Index;
  for (Index n : {55, 64, 77, 115, 128, 256, 432, 1000, 2000, 4832}) {
    auto bench = [n](auto const &_) {
      using Mat = std::remove_cv_t<std::remove_reference_t<decltype(_)>>;
      Mat a(n, n);
      Mat b(n, 4);

      a.setRandom();
      a = (a * a.adjoint()).eval();
      b.setRandom();

      std::cout << "  positive definite\n";
      {
        Eigen::LLT<Mat> dec(n);
        auto elapsed = timeit([&]() { dec.compute(a); });
        Mat x = dec.solve(b);
        std::cout << "    llt error: " << (a * x - b).cwiseAbs().maxCoeff()
                  << "\n";
        std::cout << "    time elapsed: " << elapsed << "s"
                  << "\n\n";
      }
      {
        Eigen::LDLT<Mat> dec(n);
        auto elapsed = timeit([&]() { dec.compute(a); });
        Mat x = dec.solve(b);
        std::cout << "    ldlt error: " << (a * x - b).cwiseAbs().maxCoeff()
                  << "\n";
        std::cout << "    time elapsed: " << elapsed << "s"
                  << "\n\n";
      }
      {
        Eigen::BunchKaufman<Mat> dec(n);
        auto elapsed = timeit([&]() { dec.compute(a); });
        Mat x = dec.solve(b);
        std::cout << "    bunch-kaufman error: "
                  << (a * x - b).cwiseAbs().maxCoeff() << "\n";
        std::cout << "    time elapsed: " << elapsed << "s"
                  << "\n\n";
      }
      std::cout << '\n';

      a.setRandom();
      a = (a + a.adjoint()).eval();
      b.setRandom();

      std::cout << "  indefinite\n";
      {
        Eigen::LDLT<Mat> dec(n);
        auto elapsed = timeit([&]() { dec.compute(a); });
        Mat x = dec.solve(b);
        std::cout << "    ldlt error: " << (a * x - b).cwiseAbs().maxCoeff()
                  << "\n";
        std::cout << "    time elapsed: " << elapsed << "s"
                  << "\n\n";
      }
      {
        Eigen::BunchKaufman<Mat> dec(n);
        auto elapsed = timeit([&]() { dec.compute(a); });
        Mat x = dec.solve(b);
        std::cout << "    bunch-kaufman error: "
                  << (a * x - b).cwiseAbs().maxCoeff() << "\n";
        std::cout << "    time elapsed: " << elapsed << "s"
                  << "\n\n";
      }
    };
    std::cout << "dim: " << n << "\n\n";
    std::cout << "f64" << '\n';
    bench(Eigen::MatrixXd());
    std::cout << "c64" << '\n';
    bench(Eigen::MatrixXcd());
  }
}
