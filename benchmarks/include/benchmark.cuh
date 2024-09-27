#ifndef _BENCHMARK_CUH_
#define _BENCHMARK_CUH_

#include "timeit.cuh"
#include <fstream>

#include <concepts>
#include <iostream>
#include <ostream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "indicators/progress_bar.hpp"

namespace detail {

using namespace indicators;

ProgressBar make_pbar(const char* label) {
  // clang-format off
  return ProgressBar {
    option::BarWidth{50},
    option::Start{"["},
    option::Fill{"■"},
    option::Lead{"■"},
    option::Remainder{"-"},
    option::End{" ]"},
    option::PostfixText{label},
    option::ForegroundColor{Color::cyan},
  };
  // clang-format on
}

class ProgressBarPusher {
 private:
  ProgressBar& _pbar;
  uint _curr;
  uint _total;

 public:
  ProgressBarPusher(ProgressBar& pbar, size_t curr, size_t total)
      : _pbar(pbar)
      , _curr(curr)
      , _total(total) {}

  ~ProgressBarPusher() {
    _pbar.set_option(indicators::option::PrefixText{
        std::to_string(_curr + 1) + "/" + std::to_string(_total)});
    _pbar.set_progress(static_cast<float>(_curr) / _total * 100);
    if (_curr + 1 == _total) _pbar.set_progress(100);
  }
};

template <typename T>
concept TupleLike = requires(T a) {
  std::tuple_size<T>::value;
  std::get<0>(a);
};

template <TupleLike T>
auto make_indices_for_tuple(T) {
  return std::make_index_sequence<std::tuple_size_v<T>>{};
}

}  // namespace detail

template <class J>
concept JobFunc = requires(J j) {
  { j(size_t{}) } -> detail::TupleLike;
};

template <class G>
concept GrowthFunc = requires(G g) {
  { g(size_t{}) } -> std::same_as<size_t>;
};

template <TimerKind TimerT, JobFunc JobT, GrowthFunc GrowthT, class TargetT>
class Benchmark final {
 private:
  const char* _name;
  TimeIt<TimerT, TargetT> _timeit;
  std::vector<size_t> _job_sizes;
  const JobT& _job;
  const GrowthT& _growth;

  template <class TupleT, size_t... I>
  void _apply_timeit(TupleT args_tuple, std::index_sequence<I...>) {
    _timeit.run(std::get<I>(args_tuple)...);
  }

 public:
  Benchmark(const char* name,
      TimerT,
      const JobT& job,
      const GrowthT& growth,
      const TargetT& target,
      uint nrepeats = 1,
      uint nwarmups = 0)
      : _name(name)
      , _job(job)
      , _growth(growth)
      , _timeit(target, nrepeats, nwarmups) {}

  Benchmark& run(size_t nprobes) {
    for (size_t i = 0; i < nprobes; ++i) {
      _job_sizes.push_back(_growth(i));
      auto args = _job(_job_sizes.back());
      _apply_timeit(args, detail::make_indices_for_tuple(args));
    }
    return *this;
  }

  Benchmark& run(size_t nprobes, bool) {
    auto pbar = detail::make_pbar(_name);
    for (size_t i = 0; i < nprobes; ++i) {
      detail::ProgressBarPusher bpp(pbar, i, nprobes);  // NOLINT
      _job_sizes.push_back(_growth(i));
      auto args = _job(_job_sizes.back());
      _apply_timeit(args, detail::make_indices_for_tuple(args));
    }
    return *this;
  }

  const std::vector<double>& measurements() const {
    return _timeit.measurements();
  }

  const std::vector<size_t>& job_sizes() const {
    return _job_sizes;
  }

  void reset() {
    _timeit.reset();
    _job_sizes.clear();
  }

  void export_to_csv(const char* fname = nullptr, const char* head = nullptr) const {
    auto jobs = job_sizes();

    if (jobs.empty()) return;
    auto ts = measurements();

    if (not fname) fname = _name;
    std::ofstream os(fname);

    if (head) os << head << std::endl;

    for (size_t i = 0; i < jobs.size(); ++i) {
      os << jobs[i] << ',' << ts[i] << std::endl;
    }
  }
};

#endif  // _BENCHMARK_CUH_
