#ifndef _VECVIEW_CUH_
#define _VECVIEW_CUH_

#include "dynblock.cuh"
#include "vec.cuh"

template <ItemKind T>
class VecView final {
 private:
  const DynBlock& _block;
  size_t _ori;
  size_t _len;

 public:
  struct vec_view_f {};

  using item_t = T;

  template <ItemKind U>
  VecView(const Vec<U>& vec, size_t a, size_t b)
      : _block(vec)
      , _ori(0)
      , _len(0) {

    size_t size = vec.size();
    _ori = a * sizeof(U);
    size_t last = (b - 1) * sizeof(U);
    if (_ori > last) std::swap(_ori, last);
    if (_ori >= size) _ori = 0;
    if (last >= size) last = size - 1;
    _len = (last - _ori + 1) / sizeof(T);
  }

  template <ItemKind U>
  VecView(const Vec<U>& vec)
      : _block(vec)
      , _ori(0)
      , _len(vec.size() / sizeof(T)) {}

  VecView(const Vec<T>& vec)
      : _block(vec)
      , _ori(0)
      , _len(vec.len()) {}

  VecView(const VecView& vec_view)
      : _block(vec_view._block)
      , _ori(vec_view._ori)
      , _len(vec_view._len) {}

  const T* data() const {
    return reinterpret_cast<const T*>(_block.data() + _ori);
  }

  T operator[](size_t i) const {
    return *(data() + i);
  }

  size_t len() const {
    return _len;
  }

  Loc loc() const {
    return _block.loc();
  }

  bool is_on_host() const {
    return _block.is_on_host();
  }

  bool is_on_device() const {
    return _block.is_on_device();
  }
};

#endif  // _VECVIEW_CUH_
