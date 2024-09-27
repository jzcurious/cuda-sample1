#ifndef _TKINDS_CUH_
#define _TKINDS_CUH_

#include <concepts>

template <class T>
concept ItemKind = std::floating_point<T> || std::integral<T>;

template <class T>
concept ScalType = requires { typename std::decay_t<T>::scal_f; };

template <class T>
concept VecType = requires { typename std::decay_t<T>::vec_f; };

template <class T>
concept VecViewType = requires { typename std::decay_t<T>::vec_view_f; };

template <class T>
concept VecKind = VecType<T> || VecViewType<T>;

#endif  // _TKINDS_CUH_
