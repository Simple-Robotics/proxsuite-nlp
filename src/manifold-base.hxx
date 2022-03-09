#pragma once


namespace lienlp {

  template<class T>
  template<class Vec_t, class Tangent_t>
  void ManifoldTpl<T>::integrate(
    const Eigen::MatrixBase<Vec_t>& x,
    const Eigen::MatrixBase<Tangent_t>& v,
    Eigen::MatrixBase<Vec_t>& xout) const
  {
    derived().integrate_impl(x.derived(), v.derived(), xout.derived());
  }

  template<class T>
  template<class Vec_t, class Tangent_t>
  typename ManifoldTpl<T>::Point_t
  ManifoldTpl<T>::integrate(const Eigen::MatrixBase<Vec_t>& x,
                            const Eigen::MatrixBase<Tangent_t>& v) const
  {
    Point_t out;
    out.resize(nx());
    integrate(x, v, out);
    return out;
  }

  template<class T>
  template<class Vec_t, class Tangent_t, class Jout_t>
  void ManifoldTpl<T>::Jintegrate(const Eigen::MatrixBase<Vec_t>& x,
                                  const Eigen::MatrixBase<Tangent_t>& v,
                                  Eigen::MatrixBase<Jout_t>& Jout,
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
  template<class Vec_t, class Tangent_t>
  void ManifoldTpl<T>::diff(
    const Eigen::MatrixBase<Vec_t>& x0,
    const Eigen::MatrixBase<Vec_t>& x1,
    Eigen::MatrixBase<Tangent_t>& out) const
  {
    derived().diff_impl(x0.derived(), x1.derived(), out.derived());
  }

  template<class T>
  template<class Vec_t>
  typename ManifoldTpl<T>::TangentVec_t
  ManifoldTpl<T>::diff(const Eigen::MatrixBase<Vec_t>& x0,
                       const Eigen::MatrixBase<Vec_t>& x1) const
  {
    TangentVec_t out;
    out.resize(ndx());
    diff(x0, x1, out);
    return out;
  }

  template<class T>
  template<class Vec_t, class Jout_t>
  void ManifoldTpl<T>::Jdiff(const Eigen::MatrixBase<Vec_t>& x0,
                             const Eigen::MatrixBase<Vec_t>& x1,
                             Eigen::MatrixBase<Jout_t>& Jout,
                             int arg) const
  {
    switch (arg) {
      case 0:
        Jdiff<0>(x0, x1, Jout);
        break;
      case 1:
        Jdiff<1>(x0, x1, Jout);
        break;
    }
  }

  template<class T>
  int ManifoldTpl<T>::nx() const { return derived().nx_impl(); }  /// get repr dimension
  template<class T>
  int ManifoldTpl<T>::ndx() const { return derived().ndx_impl(); }  /// get tangent space dim

}
