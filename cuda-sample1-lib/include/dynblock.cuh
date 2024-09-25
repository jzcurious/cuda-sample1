#ifndef _DYNBLOCK_CUH_
#define _DYNBLOCK_CUH_

enum Loc { Host = 0, Device = 1 };

using byte = char;

class DynBlock {
 private:
  size_t _size;
  mutable Loc _loc;
  mutable byte* _data;

  void _alloc_data() const;
  void _free_data() const;
  void _copy_data(const DynBlock& block) const;
  void _copy_data_from_host(const byte* src, size_t size);

 public:
  DynBlock(size_t size, Loc loc = Loc::Host);
  DynBlock(const void* src, size_t size, Loc loc);
  DynBlock(const DynBlock& block);
  DynBlock(DynBlock&& block);
  DynBlock& operator=(const DynBlock& block);
  DynBlock& operator=(DynBlock&& block);
  ~DynBlock();

  DynBlock& rewrite(const DynBlock& block);
  DynBlock& rewrite(DynBlock&& block);
  size_t size() const;
  Loc loc() const;
  bool is_on_host() const;
  bool is_on_device() const;
  void to(Loc loc) const;
  void to_host() const;
  void to_device() const;
  byte* data();
  const byte* data() const;
};

#endif  // _DYNBLOCK_CUH_
