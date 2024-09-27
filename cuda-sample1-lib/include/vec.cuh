#ifndef _VEC_CUH_
#define _VEC_CUH_

#include "dynblock.cuh"
#include "tkinds.cuh"

template <ItemKind T>
class Vec final : public DynBlock {
 private:
  size_t _len;

 public:
  struct vec_f {};

  using item_t = T;

  Vec(size_t len, Loc loc = Loc::Host)
      : DynBlock(len * sizeof(T), loc)
      , _len(len) {}

  Vec(const Vec& vec)
      : DynBlock(vec)
      , _len(vec._len) {}

  Vec(Vec&& vec)
      : DynBlock(std::move(vec))
      , _len(vec._len) {}

  Vec& operator=(const Vec& vec) {
    DynBlock::operator=(vec);
    _len = vec._len;
    return *this;
  }

  Vec& operator=(Vec&& vec) {
    if (this == &vec) return *this;
    DynBlock::operator=(std::move(vec));
    _len = vec._len;
    return *this;
  }

  const T* data() const {
    return reinterpret_cast<const T*>(DynBlock::data());
  }

  T* data() {
    return reinterpret_cast<T*>(DynBlock::data());
  }

  T& operator[](size_t i) {
    return *(data() + i);
  }

  T operator[](size_t i) const {
    return *(data() + i);
  }

  size_t len() const {
    return _len;
  }
};

#endif  // _VEC_CUH_
