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
    option::FontStyles{std::vector<FontStyle>{FontStyle::bold}}
  };
  // clang-format on
}

template <typename T>
concept TupleLike = requires(T a) {
  std::tuple_size<T>::value;
  std::get<0>(a);
};

}  // namespace detail

template <class J>
concept JobFunc = requires(J j) {
  { j(size_t{}) } -> detail::TupleLike;
};

template <class G>
concept GrowthFunc = requires(G g) {
  { g(size_t{}) } -> std::same_as<size_t>;
};

template <auto target, auto job, TimeItMode tmode = TimeItMode{}>
  requires JobFunc<decltype(job)>
class Benchmark final {
 private:
  TimeIt<target, tmode> _timeit;
  std::vector<size_t> _job_sizes;

  template <class TupleT, size_t... I>
  void _run_timeit_with_tuple(TupleT args_tuple, std::index_sequence<I...>) {
    _timeit.run(std::get<I>(args_tuple)...);
  }

 public:
  const char* name;

  Benchmark(const char* name)
      : name(name) {}

  template <GrowthFunc G>
  Benchmark& run(G growth, size_t nprobes) {
    for (size_t i = 0; i < nprobes; ++i) {
      auto job_size = growth(i);
      auto args = job(job_size);
      _job_sizes.push_back(job_size);
      _run_timeit_with_tuple(
          args, std::make_index_sequence<std::tuple_size_v<decltype(args)>>{});
    }
    return *this;
  }

  template <GrowthFunc G>
  Benchmark& run(G growth, size_t nprobes, bool) {
    auto pbar = detail::make_pbar(this->name);
    for (size_t i = 0; i < nprobes; ++i) {
      auto job_size = growth(i);
      auto args = job(job_size);
      _job_sizes.push_back(job_size);
      _run_timeit_with_tuple(
          args, std::make_index_sequence<std::tuple_size_v<decltype(args)>>{});
      pbar.set_option(indicators::option::PrefixText{
          std::to_string(i + 1) + "/" + std::to_string(nprobes)});
      pbar.set_progress(static_cast<float>(i) / nprobes * 100);
    }
    pbar.set_progress(100);
    return *this;
  }

  const measurements_t measurements() const {
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

    if (not fname) fname = this->name;
    std::ofstream os(fname);

    if (head) os << head << std::endl;

    for (size_t i = 0; i < jobs.size(); ++i) {
      os << jobs[i] << ',' << ts[i] << std::endl;
    }
  }
};

#endif  // _BENCHMARK_CUH_
