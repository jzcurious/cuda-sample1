#ifndef _SCAL_CUH_
#define _SCAL_CUH_

#include "dynblock.cuh"
#include "tkinds.cuh"

template <ItemKind T>
class Scal final : public DynBlock {
 public:
  struct scal_f {};

  using item_t = T;

  Scal(T val, Loc loc = Loc::Host)
      : DynBlock(&val, sizeof(T), loc) {}

  Scal(const Scal& scal)
      : DynBlock(scal) {}

  Scal(Scal&& scal)
      : DynBlock(std::move(scal)) {}

  Scal& operator=(T val) {
    DynBlock::operator=(DynBlock(&val, sizeof(T), Loc::Host));
    return *this;
  }

  Scal& operator=(const Scal& scal) {
    if (this == &scal) return *this;
    DynBlock::operator=(scal);
    return *this;
  }

  Scal& operator=(Scal&& scal) {
    if (this == &scal) return *this;
    DynBlock::operator=(std::move(scal));
    return *this;
  }

  const T* data() const {
    return reinterpret_cast<const T*>(DynBlock::data());
  }

  T* data() {
    return reinterpret_cast<T*>(DynBlock::data());
  }

  operator T() const {
    return *data();
  }

  operator T&() {
    return *data();
  }
};

#endif  // _SCAL_CUH_
