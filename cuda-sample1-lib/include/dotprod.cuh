#ifndef _DOTPROD_CUH_
#define _DOTPROD_CUH_

#include "gridconf.cuh"
#include "scal.cuh"
#include "vec.cuh"
#include "vecview.cuh"

#ifdef DEBUG
  #include <stdexcept>
#endif

#include <cuda_runtime.h>

namespace detail {

template <ItemKind T>
__global__ void dot_naive_kernel(const T* a, const T* b, T* c, size_t len) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) atomicAdd(c, a[i] * b[i]);
}

template <ItemKind T>
void dot_host(const T* a, const T* b, T* c, size_t len) {
  for (size_t i = 0; i < len; ++i) *c += a[i] * b[i];
}

}  // namespace detail

template <ItemKind T>
Scal<T> dot_views(const VecView<T>& vec_a, const VecView<T>& vec_b) {
#ifdef DEBUG
  if (vec_a.loc() != vec_b.loc())
    throw std::invalid_argument("the different location of the arguments is not allowed");
#endif

  size_t len = std::min(vec_a.len(), vec_b.len());

  if (vec_a.is_on_host()) {
    Scal<T> product(0, Loc::Host);
    detail::dot_host(vec_a.data(), vec_b.data(), product.data(), len);
    return product;
  }

  size_t grid_size = eval_grid_size(len);
  Scal<T> product(0, Loc::Device);

  detail::dot_naive_kernel<<<grid_size, block_size>>>(
      vec_a.data(), vec_b.data(), product.data(), len);
  return product;
}

template <ItemKind T>
Scal<T> dot(const Vec<T>& vec_a, const Vec<T>& vec_b) {
  return dot_views(VecView(vec_a), VecView(vec_b));
}
#endif  // _DOTPROD_CUH_
