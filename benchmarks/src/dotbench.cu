#include "argparse/argparse.hpp"
#include "benchmark.cuh"
#include "dotprod.cuh"
#include "fillvec.cuh"
#include "timers.cuh"
#include "tkinds.cuh"
#include "vec.cuh"
#include "vecview.cuh"

#include <functional>

size_t growth_linear100(size_t n, size_t bias) {
  return n * 100 + bias;
}

size_t growth_exp10(size_t n, size_t bias) {
  return std::pow(10, n) + bias;
}

template <ItemKind T>
class VecProvider final {
 private:
  Vec<T> _v1;
  Vec<T> _v2;

 public:
  VecProvider(size_t max_len, Loc loc)
      : _v1(random_vec_m1s1<float>(max_len, Loc::Device))
      , _v2(random_vec_m1s1<float>(max_len, Loc::Device)) {
    _v1.to(loc);
    _v2.to(loc);
  }

  std::tuple<VecView<T>, VecView<T>> operator()(size_t j) const {
    return {VecView<T>(_v1, 0, j + 1), VecView<T>(_v2, 0, j + 1)};
  }
};

int main(int argc, char* argv[]) {
  argparse::ArgumentParser program("dotbench");

  // clang-format off

  program.add_argument("-n", "--nprobes")
    .help("specify the number of probes")
    .default_value(100u)
    .scan<'u', unsigned>();

  program.add_argument("-o", "--output")
    .help("specify the output file")
    .default_value("dotbench.csv");

  program.add_argument("-d", "--device")
    .help("if set, compute on GPU")
    .flag();

  program.add_argument("-e", "--exp")
    .help("if set, the job is growing exponentially (base = 2)")
    .flag();

  program.add_argument("-b", "--bias")
    .help("specify the job size bias")
    .default_value(1u)
    .scan<'u', unsigned>();

  // clang-format on

  try {
    program.parse_args(argc, argv);
  } catch (const std::exception& err) {
    std::cerr << err.what() << std::endl;
    std::cerr << program;
    std::exit(1);
  }

  auto nprobes = program.get<unsigned>("--nprobes");
  auto fname = program.get<std::string>("--output");
  auto bias = program.get<unsigned>("--bias");

  std::function<size_t(size_t)> growth
      = program["--exp"] == true
            ? std::bind(&growth_exp10, std::placeholders::_1, bias)
            : std::bind(&growth_linear100, std::placeholders::_1, bias);

  Loc loc = program["--device"] == true ? Loc::Device : Loc::Host;
  VecProvider<float> vp(growth(nprobes - 1), loc);

  if (program["--device"] == true) {
    Benchmark("GPU", GPUTimer{}, vp, growth, dot_views<float>, 2)
        .run(nprobes, true)
        .export_to_csv(fname.c_str(), "Length,Time");
    return 0;
  }

  Benchmark("CPU", CPUTimer{}, vp, growth, dot_views<float>, 2)
      .run(nprobes, true)
      .export_to_csv(fname.c_str(), "Length,Time");
  return 0;
}
