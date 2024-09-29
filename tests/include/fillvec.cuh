#ifndef _FILLVEC_CUH_
#define _FILLVEC_CUH_

#include "vec.cuh"

#include <curand.h>
#include <random>

namespace detail {

template <VecKind VecT, std::uniform_random_bit_generator GenT>
void fill_normal_host(VecT& vec, GenT& gen, std::normal_distribution<>& dist) {
  for (size_t i = 0; i < vec.len(); ++i) vec[i] = dist(gen);
}

template <VecKind VecT>
void fill_normal_device(
    VecT& vec, curandGenerator_t& gen, std::normal_distribution<>& dist) {
  curandGenerateNormal(gen, vec.data(), vec.len(), dist.mean(), dist.stddev());
}

class DefaultGenerators final {
 private:
  std::mt19937 _host_gen;
  curandGenerator_t _device_gen;

 public:
  DefaultGenerators(unsigned seed)
      : _host_gen(seed) {
    curandCreateGenerator(&_device_gen, CURAND_RNG_PSEUDO_MT19937);
    curandSetPseudoRandomGeneratorSeed(_device_gen, seed);
  }

  ~DefaultGenerators() {
    curandDestroyGenerator(_device_gen);
  }

  std::mt19937& host_gen() {
    return _host_gen;
  }

  curandGenerator_t& device_gen() {
    return _device_gen;
  }
};

}  // namespace detail

template <class T>
concept GeneratorKind
    = std::uniform_random_bit_generator<T> || std::is_same_v<T, curandGenerator_t>;

template <VecKind VecT>
void fill_normal(VecT& vec, curandGenerator_t& gen, std::normal_distribution<>& dist) {
  bool located_on_host = vec.is_on_host();
  if (located_on_host) vec.to_device();
  detail::fill_normal_device(vec, gen, dist);
  if (located_on_host) vec.to_host();
}

template <VecKind VecT, std::uniform_random_bit_generator GenT>
void fill_normal(VecT& vec, GenT& gen, std::normal_distribution<>& dist) {
  bool located_on_device = vec.is_on_device();
  if (located_on_device) vec.to_host();
  detail::fill_normal_host(vec, gen, dist);
  if (located_on_device) vec.to_device();
}

template <ItemKind T>
Vec<T> random_vec_m1s1(size_t n, Loc loc = Loc::Host) {
  static std::mt19937 r;
  static detail::DefaultGenerators gens(r());
  Vec<T> vec(n, loc);
  auto dist = std::normal_distribution<>(1, 1);

  if constexpr (std::integral<T>) {
    fill_normal(vec, gens.host_gen(), dist);
  } else {
    if (vec.is_on_device())
      fill_normal(vec, gens.device_gen(), dist);
    else
      fill_normal(vec, gens.host_gen(), dist);
  }

  return vec;
}

template <ItemKind T, GeneratorKind GenT>
Vec<T> random_vec_m1s1(size_t n, GenT& gen, Loc loc = Loc::Host) {
  Vec<T> vec(n, loc);
  auto dist = std::normal_distribution<>(5, 2);
  fill_normal(vec, gen, dist);
  return vec;
}
#endif  // _FILLVEC_CUH_
