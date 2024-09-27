#include "timers.cuh"

namespace detail {
static double time_placeholder = 0;
}

CPUTimer::CPUTimer()
    : _time(detail::time_placeholder) {}

CPUTimer::CPUTimer(double& time)
    : _time(time)
    , _start(clock_t::now()) {}

CPUTimer::~CPUTimer() {
  time_point_t stop = clock_t::now();
  _time += std::chrono::duration<double>(stop - _start).count();
}

GPUTimer::GPUTimer()
    : _time(detail::time_placeholder) {}

GPUTimer::GPUTimer(double& time)
    : _time(time) {
  cudaDeviceSynchronize();
  cudaEventCreate(&_start);
  cudaEventCreate(&_stop);
  cudaEventRecord(_start);
}

GPUTimer::~GPUTimer() {
  cudaDeviceSynchronize();
  cudaEventRecord(_stop);
  cudaEventSynchronize(_stop);
  float delta;
  cudaEventElapsedTime(&delta, _start, _stop);
  cudaEventDestroy(_start);
  cudaEventDestroy(_stop);
  _time += static_cast<double>(delta) / 1000;
}
