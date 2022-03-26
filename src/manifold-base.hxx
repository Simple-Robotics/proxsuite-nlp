#pragma once


namespace lienlp
{

  template<class T>
  template<class Vec_t, class Tangent_t, class Out_t>
  void ManifoldTpl<T>::integrate(
    const Eigen::MatrixBase<Vec_t>& x,
    const Eigen::MatrixBase<Tangent_t>& v,
    const Eigen::MatrixBase<Out_t>& xout) const
  {
    derived().integrate_impl(x.derived(), v.derived(), xout.derived());
  }

  template<class T>
  template<int arg, class Vec_t, class Tangent_t, class Jout_t>
  void ManifoldTpl<T>::Jintegrate(
    const Eigen::MatrixBase<Vec_t>& x,
    const Eigen::MatrixBase<Tangent_t>& v,
    const Eigen::MatrixBase<Jout_t>& Jout) const
  {
    derived().template Jintegrate_impl<arg>(x.derived(), v.derived(), Jout.derived());
  }

  template<class T>
  template<class Vec_t, class Tangent_t, class Jout_t>
  void ManifoldTpl<T>::Jintegrate(const Eigen::MatrixBase<Vec_t>& x,
                                  const Eigen::MatrixBase<Tangent_t>& v,
                                  const Eigen::MatrixBase<Jout_t>& Jout,
                                  int arg) const
  {
    switch (arg)
    {
      case 0:
        Jintegrate<0>(x, v, Jout);
        break;
      case 1:
        Jintegrate<1>(x, v, Jout);
        break;
    }
  }

  // Difference operators


  template<class T>
  template<class Vec1_t, class Vec2_t, class Out_t>
  void ManifoldTpl<T>::difference(
    const Eigen::MatrixBase<Vec1_t>& x0,
    const Eigen::MatrixBase<Vec2_t>& x1,
    const Eigen::MatrixBase<Out_t>& out) const
  {
    derived().difference_impl(x0.derived(), x1.derived(), out.derived());
  }

  template<class T>
  template<int arg, class Vec1_t, class Vec2_t, class Jout_t>
  void ManifoldTpl<T>::Jdifference(
    const Eigen::MatrixBase<Vec1_t>& x0,
    const Eigen::MatrixBase<Vec2_t>& x1,
    const Eigen::MatrixBase<Jout_t>& Jout) const
  {
    derived().template Jdifference_impl<arg>(x0.derived(), x1.derived(), Jout.derived());
  }

  template<class T>
  template<class Vec1_t, class Vec2_t, class Jout_t>
  void ManifoldTpl<T>::Jdifference(
    const Eigen::MatrixBase<Vec1_t>& x0,
    const Eigen::MatrixBase<Vec2_t>& x1,
    const Eigen::MatrixBase<Jout_t>& Jout,
    int arg) const
  {
    switch (arg) {
      case 0:
        Jdifference<0>(x0, x1, Jout);
        break;
      case 1:
        Jdifference<1>(x0, x1, Jout);
        break;
    }
  }

  template<class T>
  int ManifoldTpl<T>::nx() const { return derived().nx_impl(); }  /// get repr dimension
  template<class T>
  int ManifoldTpl<T>::ndx() const { return derived().ndx_impl(); }  /// get tangent space dim

  /// Allocated versions
  template<class T>
  template<class Vec_t, class Tangent_t>
  typename ManifoldTpl<T>::PointType
  ManifoldTpl<T>::integrate(
    const Eigen::MatrixBase<Vec_t>& x,
    const Eigen::MatrixBase<Tangent_t>& v) const
  {
    PointType out;
    out.resize(nx());
    integrate(x, v, out);
    return out;
  }

  template<class T>
  template<class Vec1_t, class Vec2_t>
  typename ManifoldTpl<T>::TangentVectorType
  ManifoldTpl<T>::difference(
    const Eigen::MatrixBase<Vec1_t>& x0,
    const Eigen::MatrixBase<Vec2_t>& x1) const
  {
    TangentVectorType out;
    out.resize(ndx());
    difference(x0, x1, out);
    return out;
  }


}
