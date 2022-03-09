
/// Macro typedefs for dynamic-sized vectors/matrices, used for cost funcs, merit funcs
/// because we don't CRTP them and virtual members funcs can't be templated.
#define LIENLP_DEFINE_DYNAMIC_TYPES(_Scalar)                \
  using Scalar = _Scalar;                                   \
  using VectorXs = typename math_types<Scalar>::VectorXs;   \
  using MatrixXs = typename math_types<Scalar>::MatrixXs;   \
  using RefVector = Eigen::Ref<const VectorXs>;             \
  using RefMatrix = Eigen::Ref<const MatrixXs>;

/// @brief Macro empty arg
#define LIENLP_MACRO_EMPTY_ARG

#define LIENLP_EIGEN_CONST_CAST(type, obj) const_cast<type &>(obj.derived())
