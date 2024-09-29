#include "dotprod.cuh"
#include "fillvec.cuh"

#include "gtest/gtest.h"

#define __alloc_test_scal(_type)                                                         \
  TEST(AllocTest, _type##Scal) {                                                         \
    auto s1 = Scal<_type>(123, Loc::Host);                                               \
    auto s2 = Scal<_type>(123, Loc::Device);                                             \
    EXPECT_EQ(s1.loc(), Loc::Host);                                                      \
    EXPECT_EQ(s2.loc(), Loc::Device);                                                    \
  }

#define __alloc_test_vec(_type, _len)                                                    \
  TEST(AllocTest, _type##Vec##Len##_len) {                                               \
    auto v1 = Vec<_type>(_len, Loc::Host);                                               \
    auto v2 = Vec<_type>(_len, Loc::Device);                                             \
    EXPECT_EQ(v1.loc(), Loc::Host);                                                      \
    EXPECT_EQ(v2.loc(), Loc::Device);                                                    \
  }

#define __trans_test_scal(_type)                                                         \
  TEST(TransTest, _type##Scal) {                                                         \
    auto s1 = Scal<_type>(123, Loc::Host);                                               \
    auto s2 = Scal<_type>(123, Loc::Device);                                             \
    s1 = 999;                                                                            \
    s2 = s1;                                                                             \
    s1.to_device();                                                                      \
    s1.to_host();                                                                        \
    s2.to_host();                                                                        \
    s2.to_device();                                                                      \
    s2.to_host();                                                                        \
    EXPECT_EQ(s1, s2);                                                                   \
  }

#define __trans_test_vec(_type, _len)                                                    \
  TEST(TransTest, _type##Vec##Len##_len) {                                               \
    auto v1 = random_vec_m1s1<_type>(_len);                                              \
    auto v2 = Vec<_type>(_len, Loc::Device);                                             \
    v2 = v1;                                                                             \
    v1.to_device();                                                                      \
    v1.to_host();                                                                        \
    v2.to_host();                                                                        \
    v2.to_device();                                                                      \
    v2.to_host();                                                                        \
    EXPECT_EQ(v1[0], v2[0]);                                                             \
    EXPECT_EQ(v1[_len / 2], v2[_len / 2]);                                               \
    EXPECT_EQ(v1[_len - 1], v2[_len - 1]);                                               \
  }

#define __dot_test(_type, _len, _abs)                                                    \
  TEST(DotTest, _type##Len##_len) {                                                      \
    auto v1 = random_vec_m1s1<_type>(_len);                                              \
    auto v2 = random_vec_m1s1<_type>(_len);                                              \
    auto p1 = dot(v1, v2);                                                               \
    v1.to_device();                                                                      \
    v2.to_device();                                                                      \
    auto p2 = dot(v1, v2);                                                               \
    p2.to_host();                                                                        \
    EXPECT_NEAR(p1, p2, _abs);                                                           \
  }

#define __test_item_type(_type)                                                          \
  __alloc_test_scal(_type);                                                              \
  __alloc_test_vec(_type, 101);                                                          \
  __trans_test_scal(_type);                                                              \
  __trans_test_vec(_type, 101);                                                          \
  __trans_test_vec(_type, 1001);                                                         \
  __trans_test_vec(_type, 10001);                                                        \
  __dot_test(_type, 1, 0.05);                                                            \
  __dot_test(_type, 101, 0.05);                                                          \
  __dot_test(_type, 1001, 0.5);                                                          \
  __dot_test(_type, 10001, 0.5);

__test_item_type(float);
__test_item_type(int);
__test_item_type(unsigned);

#if __CUDA_ARCH__ >= 800
__test_item_type(double);
#endif
