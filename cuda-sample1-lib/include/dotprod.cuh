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

// template <size_t persist_coeff, ItemKind T>
// __global__ void dot_persistent_kernel(const T* a, const T* b, T* c, size_t len) {
//   size_t from = persist_coeff * (blockIdx.x * blockDim.x + threadIdx.x);
//   size_t to = from + persist_coeff;

//   T acc = 0;
//   for (size_t i = from; i < to and i < len; ++i) acc += a[i] * b[i];
//   atomicAdd(c, acc);
// }

template <size_t persist_coeff, ItemKind T>
__global__ void dot_persistent_kernel(const T* a, const T* b, T* c, size_t len) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;

  T acc = 0;
  for (size_t j = 0; j < persist_coeff; ++j) {
    size_t k = i + j * persist_coeff;
    if (k < len) acc += a[k] * b[k];
  }
  atomicAdd(c, acc);
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

  size_t grid_size = eval_grid_size_persist(len);
  Scal<T> product(0, Loc::Device);

  detail::dot_persistent_kernel<persist_coeff>
      <<<grid_size, block_size>>>(vec_a.data(), vec_b.data(), product.data(), len);
  return product;
}

template <ItemKind T>
Scal<T> dot(const Vec<T>& vec_a, const Vec<T>& vec_b) {
  return dot_views(VecView(vec_a), VecView(vec_b));
}
#endif  // _DOTPROD_CUH_
