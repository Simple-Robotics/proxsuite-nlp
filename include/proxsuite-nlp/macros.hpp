/// @file
/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#pragma once

/// @brief Macro empty arg
#define PROXSUITE_NLP_MACRO_EMPTY_ARG

#define PROXSUITE_NLP_EIGEN_CONST_CAST(type, obj) const_cast<type &>(obj)

#ifdef PROXSUITE_NLP_EIGEN_CHECK_MALLOC
#define PROXSUITE_NLP_EIGEN_ALLOW_MALLOC(allowed)                              \
  ::Eigen::internal::set_is_malloc_allowed(allowed)
#else
#define PROXSUITE_NLP_EIGEN_ALLOW_MALLOC(allowed)
#endif

/// @brief Entering performance-critical code.
#define PROXSUITE_NLP_NOMALLOC_BEGIN PROXSUITE_NLP_EIGEN_ALLOW_MALLOC(false)

/// @brief Exiting performance-critical code.
#define PROXSUITE_NLP_NOMALLOC_END PROXSUITE_NLP_EIGEN_ALLOW_MALLOC(true)

#ifdef __GNUC__
#define PROXSUITE_NLP_INLINE inline __attribute__((always_inline))
#else
#define PROXSUITE_NLP_INLINE inline
#endif

#if defined _WIN32 || defined __CYGWIN__
#define PROXSUITE_NLP_DLLIMPORT_EXTERN extern
#define PROXSUITE_NLP_DLLEXPORT_EXTERN
#else
#define PROXSUITE_NLP_DLLIMPORT_EXTERN extern
#define PROXSUITE_NLP_DLLEXPORT_EXTERN extern
#endif

#ifdef proxsuite_nlp_EXPORTS
#define PROXSUITE_NLP_EXTERN PROXSUITE_NLP_DLLEXPORT_EXTERN
#else
#define PROXSUITE_NLP_EXTERN PROXSUITE_NLP_DLLIMPORT_EXTERN
#endif
