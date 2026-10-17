/* SPDX-License-Identifier: MIT */
/* Copyright (c) 2026 The Colibri Authors */
#ifndef CLB_EXPORT_H
#define CLB_EXPORT_H

/* ---- 版本 ---- */
#define CLB_VERSION_MAJOR 0
#define CLB_VERSION_MINOR 1
#define CLB_VERSION_PATCH 0
#define CLB_VERSION_STRING "0.1.0"

/* ---- 符号可见性 ---- */
#if defined(_WIN32) || defined(__CYGWIN__)
#  if defined(CLB_BUILD_SHARED)
#    define CLB_API __declspec(dllexport)
#  elif defined(CLB_USE_SHARED)
#    define CLB_API __declspec(dllimport)
#  else
#    define CLB_API
#  endif
#  define CLB_INTERNAL
#else
#  if defined(__GNUC__) && __GNUC__ >= 4
#    define CLB_API      __attribute__((visibility("default")))
#    define CLB_INTERNAL __attribute__((visibility("hidden")))
#  else
#    define CLB_API
#    define CLB_INTERNAL
#  endif
#endif

/* ---- C++ 兼容 ---- */
#ifdef __cplusplus
#  define CLB_EXTERN_C_BEGIN extern "C" {
#  define CLB_EXTERN_C_END   }
#else
#  define CLB_EXTERN_C_BEGIN
#  define CLB_EXTERN_C_END
#endif

/* ---- 编译器提示 ---- */
#if defined(__GNUC__) || defined(__clang__)
#  define CLB_NODISCARD       __attribute__((warn_unused_result))
#  define CLB_DEPRECATED(m)   __attribute__((deprecated(m)))
#  define CLB_PRINTF_FMT(f,a) __attribute__((format(printf, f, a)))
#  define CLB_LIKELY(x)       __builtin_expect(!!(x), 1)
#  define CLB_UNLIKELY(x)     __builtin_expect(!!(x), 0)
#  define CLB_INLINE          static inline __attribute__((always_inline))
#else
#  define CLB_NODISCARD
#  define CLB_DEPRECATED(m)
#  define CLB_PRINTF_FMT(f,a)
#  define CLB_LIKELY(x)   (x)
#  define CLB_UNLIKELY(x) (x)
#  define CLB_INLINE      static inline
#endif

#define CLB_UNUSED(x) ((void)(x))

#endif /* CLB_EXPORT_H */