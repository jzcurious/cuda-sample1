#ifndef _DOTPROD_CUH_
#define _DOTPROD_CUH_

#include "gridconf.cuh"
#include "scal.cuh"
#include "vec.cuh"

#include <cuda_runtime.h>

namespace detail {

template <class T>
__global__ void dot_naive_kernel(const T* a, const T* b, T* c, size_t len) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) atomicAdd(c, a[i] * b[i]);
}

template <class T>
void dot_host(const T* a, const T* b, T* c, size_t len) {
  for (size_t i = 0; i < len; ++i) *c += a[i] * b[i];
}

}  // namespace detail

template <ItemT T>
Scal<T> dot(const Vec<T>& vec_a, const Vec<T>& vec_b) {
  size_t len = std::min(vec_a.len(), vec_b.len());

  if (vec_a.is_on_host() and vec_b.is_on_host()) {
    Scal<T> product(0, Loc::Host);
    detail::dot_host(vec_a.data(), vec_b.data(), product.data(), len);
    return product;
  }

  size_t grid_size = eval_grid_size(len);
  Scal<T> product(0, Loc::Device);
  const Vec<T>* moved_to_device = nullptr;

  if (vec_a.is_on_host()) {
    vec_a.to(Loc::Device);
    moved_to_device = &vec_a;
  }

  if (vec_b.is_on_host()) {
    vec_b.to(Loc::Device);
    moved_to_device = &vec_b;
  }

  detail::dot_naive_kernel<<<grid_size, block_size>>>(
      vec_a.data(), vec_b.data(), product.data(), len);

  if (moved_to_device != nullptr) moved_to_device->to(Loc::Host);

  return product;
}

#endif  // _DOTPROD_CUH_
