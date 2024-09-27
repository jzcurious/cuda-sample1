#ifndef _TIMEIT_CUH_
#define _TIMEIT_CUH_

#include "timers.cuh"

#include <sys/types.h>
#include <vector>

template <TimerKind TimerT, class TargetT>
class TimeIt final {
 private:
  const TargetT& _target;
  const uint _nwarmups;
  const uint _nrepeats;
  std::vector<double> _measurements;

 public:
  TimeIt(const TargetT& target, uint nrepeats = 1, uint nwarmups = 0)
      : _target(target)
      , _nwarmups(nwarmups)
      , _nrepeats(nrepeats) {}

  template <class... ArgTs>
    requires(std::invocable<TargetT, ArgTs...>)
  double run(ArgTs&&... args) {
    for (uint i = 0; i < _nwarmups; ++i) _target(std::forward<ArgTs>(args)...);

    double total = 0;
    for (uint i = 0; i < _nrepeats; ++i) {
      TimerT t(total);  // NOLINT
      _target(std::forward<ArgTs>(args)...);
    }

    _measurements.push_back(total / _nrepeats);
    return last();
  }

  double last() const {
    if (_measurements.empty()) return -1.0f;
    return _measurements.back();
  }

  const std::vector<double>& measurements() const {
    return _measurements;
  }

  void reset() {
    _measurements.clear();
  }
};

#endif  // _TIMEIT_CUH_
