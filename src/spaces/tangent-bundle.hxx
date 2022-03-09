
namespace lienlp {

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

} // namespace lienlp