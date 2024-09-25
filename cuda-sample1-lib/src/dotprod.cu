#include "dotprod.cuh"  // IWYU pragma: keep

#define __register_impl(_t)                                                              \
  template Scal<_t> dot(const Vec<_t>& vec_a, const Vec<_t>& vec_b);

__register_impl(float);
__register_impl(int);
__register_impl(unsigned);

#if __CUDA_ARCH__ >= 600
__register_impl(double);
#endif

#undef __register_impl
