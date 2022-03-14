
/// Macro typedefs for dynamic-sized vectors/matrices, used for cost funcs, merit funcs
/// because we don't CRTP them and virtual members funcs can't be templated.
#define LIENLP_DEFINE_DYNAMIC_TYPES(Scalar)                   \
  using VectorXs = typename math_types<Scalar>::VectorXs;     \
  using MatrixXs = typename math_types<Scalar>::MatrixXs;     \
  using VectorOfVectors = typename math_types<Scalar>::VectorOfVectors; \
  using RefVector = Eigen::Ref<VectorXs>;                     \
  using RefMatrix = Eigen::Ref<MatrixXs>;                     \
  using ConstVectorRef = Eigen::Ref<const VectorXs>;          \
  using ConstMatrixRef = Eigen::Ref<const MatrixXs>;

/// @brief Macro empty arg
#define LIENLP_MACRO_EMPTY_ARG

#define LIENLP_EIGEN_CONST_CAST(type, obj) const_cast<type &>(obj.derived())
