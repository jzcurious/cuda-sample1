#ifndef _ITEMT_CUH_
#define _ITEMT_CUH_

template <class T>
concept ItemT = std::floating_point<T> || std::integral<T>;

#endif  // _ITEMT_CUH_
