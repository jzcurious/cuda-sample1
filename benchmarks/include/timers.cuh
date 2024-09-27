#ifndef _TIMERS_CUH_
#define _TIMERS_CUH_

#include <chrono>
#include <cuda_runtime.h>

template <class T>
concept TimerKind = requires { typename T::timer_f; };

class CPUTimer final {
  using clock_t = std::chrono::high_resolution_clock;
  using time_point_t = std::chrono::time_point<clock_t>;

 private:
  double& _time;
  time_point_t _start;

 public:
  struct timer_f {};

  CPUTimer();
  CPUTimer(double& time);
  ~CPUTimer();
};

class GPUTimer final {
 private:
  double& _time;
  cudaEvent_t _start;
  cudaEvent_t _stop;

 public:
  struct timer_f {};

  GPUTimer();
  GPUTimer(double& time);
  ~GPUTimer();
};

#endif  // _TIMERS_CUH_
