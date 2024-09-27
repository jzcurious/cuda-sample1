#ifndef _GRIDCONF_CUH_
#define _GRIDCONF_CUH_

constexpr const size_t block_size = 128;
constexpr const size_t persist_coeff = 128;

constexpr size_t eval_grid_size(size_t parallel_jobs) {
  return (parallel_jobs + block_size - 1) / block_size;
}

constexpr size_t eval_grid_size_persist(size_t parallel_jobs) {
  return (parallel_jobs + block_size * persist_coeff - 1) / (block_size * persist_coeff);
}

#endif  // _GRIDCONF_CUH_
