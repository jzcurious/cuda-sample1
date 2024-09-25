#ifndef _TIMEIT_CUH_
#define _TIMEIT_CUH_

#include <chrono>
#include <concepts>
#include <cuda_runtime.h>
#include <sys/types.h>
#include <vector>

struct TimeItMode final {
  uint repeat = 3;
  uint warmup = 1;
  bool cuda_used = false;
  bool cuda_only = false;
};

using measurements_t = std::vector<double>;

template <auto target, TimeItMode mode = TimeItMode{}>
class TimeIt final {
 private:
  measurements_t _measurements;

  using target_t = decltype(target);

 public:
  void reset() {
    _measurements.clear();
  }

  template <class... ArgTs>
    requires(std::invocable<target_t, ArgTs...> && not mode.cuda_only)
  double run(ArgTs&&... args) {
    double total = 0;

    if constexpr (mode.warmup)
      for (uint i = 0; i < mode.warmup; ++i) target(std::forward<ArgTs>(args)...);

    for (uint i = 0; i < mode.repeat; ++i) {
      if constexpr (mode.cuda_used) cudaDeviceSynchronize();
      const auto start = std::chrono::high_resolution_clock::now();
      target(std::forward<ArgTs>(args)...);
      if constexpr (mode.cuda_used) cudaDeviceSynchronize();
      const auto end = std::chrono::high_resolution_clock::now();
      total += std::chrono::duration<double>(end - start).count();  // s
    }

    _measurements.push_back(total / mode.repeat);
    return last();
  }

  template <class... ArgTs>
    requires(std::invocable<target_t, ArgTs...> && mode.cuda_only)
  double run(ArgTs&&... args) {
    double total = 0;

    float diff;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    if constexpr (mode.warmup)
      for (uint i = 0; i < mode.warmup; ++i) target(std::forward<ArgTs>(args)...);

    for (uint i = 0; i < mode.repeat; ++i) {
      cudaEventRecord(start);
      target(std::forward<ArgTs>(args)...);
      cudaEventRecord(end);
      cudaEventSynchronize(end);
      cudaEventElapsedTime(&diff, start, end);  // ms
      total += static_cast<double>(diff) / 1000;
    }

    _measurements.push_back(total / mode.repeat);
    return last();
  }

  double last() const {
    if (_measurements.empty()) return -1.0f;
    return _measurements.back();
  }

  const measurements_t& measurements() const {
    return _measurements;
  }
};

#endif  // _TIMEIT_CUH_
