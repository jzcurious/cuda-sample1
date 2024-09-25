#include "dynblock.cuh"

void DynBlock::_alloc_data() const {
  if (_data != nullptr) return;
  if (is_on_host()) {
    _data = new byte[_size];
    // cudaMallocHost(&_data, _size);
  } else {
    cudaMalloc(&_data, _size);
  }
}

void DynBlock::_free_data() const {
  if (_data == nullptr) return;
  if (is_on_host()) delete[] _data;
  // cudaFreeHost(_data);
  else
    cudaFree(_data);
  _data = nullptr;
}

void DynBlock::_copy_data(const DynBlock& block) const {
  if (_size != block._size) return;
  if (is_on_host() and block.is_on_host()) {
    std::copy(block._data, block._data + block._size, _data);
    // cudaMemcpy(_data, block._data, block._size, cudaMemcpyHostToHost);
    return;
  }
  if (is_on_host() and block.is_on_device()) {
    cudaMemcpy(_data, block._data, block._size, cudaMemcpyDeviceToHost);
    return;
  }
  if (is_on_device() and block.is_on_host()) {
    cudaMemcpy(_data, block._data, block._size, cudaMemcpyHostToDevice);
    return;
  }
  if (is_on_device() and block.is_on_device()) {
    cudaMemcpy(_data, block._data, block._size, cudaMemcpyDeviceToDevice);
    return;
  }
}

void DynBlock::_copy_data_from_host(const byte* src, size_t size) {
  if (_size != size) return;
  if (is_on_host()) {
    std::copy(src, src + size, _data);
    // cudaMemcpy(_data, src, size, cudaMemcpyHostToHost);
    return;
  }
  cudaMemcpy(_data, src, size, cudaMemcpyHostToDevice);
}

DynBlock::DynBlock(size_t size, Loc loc)
    : _size(size)
    , _loc(loc)
    , _data(nullptr) {
  _alloc_data();
}

DynBlock::DynBlock(const void* src, size_t size, Loc loc)
    : _size(size)
    , _loc(loc)
    , _data(nullptr) {
  _alloc_data();
  _copy_data_from_host(reinterpret_cast<const byte*>(src), size);
}

DynBlock::DynBlock(const DynBlock& block)
    : _size(block._size)
    , _loc(block._loc)
    , _data(nullptr) {
  _alloc_data();
  _copy_data(block);
}

DynBlock::DynBlock(DynBlock&& block)
    : _size(block._size)
    , _loc(block._loc)
    , _data(nullptr) {
  _data = block._data;
  block._data = nullptr;
}

DynBlock& DynBlock::rewrite(const DynBlock& block) {
  if (this == &block) return *this;
  if (_size != block._size) {
    _size = block._size;
    _free_data();
    _alloc_data();
  }
  _copy_data(block);
  return *this;
}

DynBlock& DynBlock::rewrite(DynBlock&& block) {
  if (this == &block) return *this;
  if (_loc == block._loc) {
    _free_data();
    _data = block._data;
    block._data = nullptr;
    return *this;
  }
  return *this = block;
}

DynBlock& DynBlock::operator=(const DynBlock& block) {
  return rewrite(block);
}

DynBlock& DynBlock::operator=(DynBlock&& block) {
  return rewrite(std::move(block));
}

DynBlock::~DynBlock() {
  _free_data();
}

bool DynBlock::is_on_host() const {
  return _loc == Loc::Host;
}

bool DynBlock::is_on_device() const {
  return _loc == Loc::Device;
}

size_t DynBlock::size() const {
  return _size;
}

Loc DynBlock::loc() const {
  return _loc;
}

void DynBlock::to(Loc loc) const {
  if (loc == _loc) return;
  auto tmp = DynBlock(std::move(*const_cast<DynBlock*>(this)));
  _loc = loc;
  _alloc_data();
  _copy_data(tmp);
}

void DynBlock::to_host() const {
  to(Loc::Host);
}

void DynBlock::to_device() const {
  to(Loc::Device);
}

byte* DynBlock::data() {
  return _data;
}

const byte* DynBlock::data() const {
  return _data;
}
