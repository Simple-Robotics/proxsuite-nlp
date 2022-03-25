
namespace lienlp
{

    template<class Base>
    typename TangentBundle<Base>::Point_t
    TangentBundle<Base>::zero_impl() const
    {
      Point_t out;
      out.resize(nx_impl());
      out.setZero();
      out.head(m_base.nx()) = m_base.zero();
      return out;
    }

    template<class Base>
    typename TangentBundle<Base>::Point_t
    TangentBundle<Base>::rand_impl() const
    {
      Point_t out;
      out.resize(nx_impl());
      out.head(m_base.nx()) = m_base.rand();
      using BTanVec_t = typename Base::TangentVec_t;
      out.tail(m_base.ndx()) = BTanVec_t::Random(m_base.ndx());
      return out;
    }

    /// Operators
    template<class Base>
    template<class Vec_t, class Tangent_t, class Out_t>
    void TangentBundle<Base>::
    integrate_impl(const Eigen::MatrixBase<Vec_t>& x,
                   const Eigen::MatrixBase<Tangent_t>& dx,
                   const Eigen::MatrixBase<Out_t>& out) const
    {
      const int nv_ = m_base.ndx();
      Out_t& out_ = LIENLP_EIGEN_CONST_CAST(Out_t, out);
      out_.resize(nx_impl());
      m_base.integrate(
        getBasePoint(x),
        getBaseTangent(dx),
        getBasePointWrite(out));
      out_.tail(nv_) = x.tail(nv_) + dx.tail(nv_);
    }

    template<class Base>
    template<class Vec1_t, class Vec2_t, class Out_t>
    void TangentBundle<Base>::
    difference_impl(const Eigen::MatrixBase<Vec1_t>& x0,
                    const Eigen::MatrixBase<Vec2_t>& x1,
                    const Eigen::MatrixBase<Out_t>& out) const
    {
      const int nv_ = m_base.ndx();
      Out_t& out_ = LIENLP_EIGEN_CONST_CAST(Out_t, out);
      out_.resize(ndx_impl());
      m_base.difference(
        getBasePoint(x0),
        getBasePoint(x1),
        getTangentHeadWrite(out));
      out_.tail(nv_) = x1.tail(nv_) - x0.tail(nv_);
    }

    template<class Base>
    template<int arg, class Vec_t, class Tangent_t, class Jout_t>
    void TangentBundle<Base>::
    Jintegrate_impl(const Eigen::MatrixBase<Vec_t>& x,
                    const Eigen::MatrixBase<Tangent_t>& dx,
                    const Eigen::MatrixBase<Jout_t>& Jout) const
    {
      const int ndxbase = m_base.ndx();
      Jout_t& J_ = LIENLP_EIGEN_CONST_CAST(Jout_t, Jout);
      J_.resize(ndx_impl(), ndx_impl());
      J_.setZero();
      m_base.template Jintegrate<arg>(
        getBasePoint(x), getBaseTangent(dx),
        getBaseJacobian(J_));
      J_.bottomRightCorner(ndxbase, ndxbase).setIdentity();
    }

    template<class Base>
    template<int arg, class Vec1_t, class Vec2_t, class Jout_t>
    void TangentBundle<Base>::
    Jdifference_impl(const Eigen::MatrixBase<Vec1_t>& x0,
                     const Eigen::MatrixBase<Vec2_t>& x1,
                     const Eigen::MatrixBase<Jout_t>& Jout) const
    {
      const int ndxbase = m_base.ndx();
      Jout_t& J_ = LIENLP_EIGEN_CONST_CAST(Jout_t, Jout);
      J_.resize(ndx_impl(), ndx_impl());
      J_.setZero();
      m_base.template Jdifference<arg>(
        getBasePoint(x0), getBasePoint(x1),
        getBaseJacobian(J_));
      if (arg == 0)
      {
        J_.bottomRightCorner(ndxbase,ndxbase).diagonal().array() = Scalar(-1);
      } else if (arg == 1) {
        J_.bottomRightCorner(ndxbase,ndxbase).setIdentity();
      }
    }

} // namespace lienlp