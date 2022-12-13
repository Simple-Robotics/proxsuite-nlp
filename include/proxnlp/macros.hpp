/// @file
/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#pragma once

/// @brief Macro empty arg
#define PROXNLP_MACRO_EMPTY_ARG

#define PROXNLP_EIGEN_CONST_CAST(type, obj) const_cast<type &>(obj)

#ifdef PROXNLP_EIGEN_CHECK_MALLOC
#define PROXNLP_EIGEN_ALLOW_MALLOC(allowed)                                    \
  ::Eigen::internal::set_is_malloc_allowed(allowed)
#else
#define PROXNLP_EIGEN_ALLOW_MALLOC(allowed)
#endif

#define PROXNLP_INLINE inline __attribute__((always_inline))
