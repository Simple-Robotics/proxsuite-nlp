/// @file
/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#pragma once

#include "proxsuite-nlp/deprecated.hpp"

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

/// \brief macros for pragma push/pop/ignore deprecated warnings
#if defined(__GNUC__) || defined(__clang__)
#define PROXSUITE_NLP_COMPILER_DIAGNOSTIC_PUSH                                 \
  PROXSUITE_NLP_PRAGMA(GCC diagnostic push)
#define PROXSUITE_NLP_COMPILER_DIAGNOSTIC_POP                                  \
  PROXSUITE_NLP_PRAGMA(GCC diagnostic pop)
#if defined(__clang__)
#define PROXSUITE_NLP_COMPILER_DIAGNOSTIC_IGNORED_DELETE_NON_ABSTRACT_NON_VIRTUAL_DTOR
PROXSUITE_NLP_PRAGMA(GCC diagnostic ignored
                     "-Wdelete-non-abstract-non-virtual-dtor")
#else
#define PROXSUITE_NLP_COMPILER_DIAGNOSTIC_IGNORED_DELETE_NON_ABSTRACT_NON_VIRTUAL_DTOR
#endif
#elif defined(WIN32)
#define PROXSUITE_NLP_COMPILER_DIAGNOSTIC_PUSH _Pragma("warning(push)")
#define PROXSUITE_NLP_COMPILER_DIAGNOSTIC_POP _Pragma("warning(pop)")
#define PROXSUITE_NLP_COMPILER_DIAGNOSTIC_IGNORED_DELETE_NON_ABSTRACT_NON_VIRTUAL_DTOR
#else
#define PROXSUITE_NLP_COMPILER_DIAGNOSTIC_PUSH
#define PROXSUITE_NLP_COMPILER_DIAGNOSTIC_POP
#define PROXSUITE_NLP_COMPILER_DIAGNOSTIC_IGNORED_DEPRECECATED_DECLARATIONS
#define PROXSUITE_NLP_COMPILER_DIAGNOSTIC_IGNORED_VARIADIC_MACROS
#define PROXSUITE_NLP_COMPILER_DIAGNOSTIC_IGNORED_SELF_ASSIGN_OVERLOADED
#define PROXSUITE_NLP_COMPILER_DIAGNOSTIC_IGNORED_MAYBE_UNINITIALIZED
#endif // __GNUC__ || __clang__
