#include "argparse/argparse.hpp"
#include "benchmark.cuh"
#include "dotprod.cuh"
#include "fillvec.cuh"
#include "itemt.cuh"
#include "vec.cuh"

#include <functional>

size_t growth_linear100(size_t n, size_t bias) {
  return n * 100 + bias;
}

size_t growth_exp10(size_t n, size_t bias) {
  return 2 << n + bias;
}

template <Loc loc, ItemT T>
std::tuple<Vec<T>, Vec<T>> gen_vectors(size_t len) {
  return {random_vec_m5s2<T>(len, loc), random_vec_m5s2<T>(len, loc)};
}

using dot_bench_host_t = Benchmark<&dot<float>, &gen_vectors<Loc::Host, float>>;

using dot_bench_device_t = Benchmark<&dot<float>,
    &gen_vectors<Loc::Device, float>,
    TimeItMode{.cuda_only = true}>;

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

  if (program["--device"] == true) {
    dot_bench_device_t{"Device"}
        .run(growth, nprobes, "Device")
        .export_to_csv(fname.c_str(), "Length,Time");
    return 0;
  }

  dot_bench_host_t{"Host"}
      .run(growth, nprobes, "Host")
      .export_to_csv(fname.c_str(), "Length,Time");
  return 0;
}
