#pragma once


namespace lienlp {

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
  template<class P_t>
  typename ManifoldTpl<T>::TangentVec_t
  ManifoldTpl<T>::diff(const Eigen::MatrixBase<P_t>& x0,
                       const Eigen::MatrixBase<P_t>& x1) const
  {
    TangentVec_t out;
    out.resize(ndx());
    diff(x0, x1, out);
    return out;
  }


  template<class T>
  int ManifoldTpl<T>::nx() const { return derived().nx_impl(); }  /// get repr dimension
  template<class T>
  int ManifoldTpl<T>::ndx() const { return derived().ndx_impl(); }  /// get tangent space dim

}
