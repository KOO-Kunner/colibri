/* SPDX-License-Identifier: MIT */
#ifndef CLB_ALLOC_H
#define CLB_ALLOC_H

#include <stddef.h>
#include <stdbool.h>
#include "clb_export.h"

CLB_EXTERN_C_BEGIN

#define CLB_DEFAULT_ALIGN 64u

CLB_API void *clb_malloc(size_t size);
CLB_API void *clb_calloc(size_t count, size_t size);
CLB_API void *clb_realloc(void *ptr, size_t size);
CLB_API void  clb_free(void *ptr);

CLB_API void *clb_aligned_alloc(size_t alignment, size_t size);
CLB_API void  clb_aligned_free(void *ptr);

/* 溢出安全乘法 / 加法 */
CLB_API bool clb_size_mul(size_t a, size_t b, size_t *out);
CLB_API bool clb_size_add(size_t a, size_t b, size_t *out);

/* ---------- Arena（线性分配器，用于张量元数据/临时区） ---------- */
typedef struct clb_arena clb_arena;

CLB_API clb_arena *clb_arena_create(size_t capacity);
CLB_API void       clb_arena_destroy(clb_arena *a);
CLB_API void      *clb_arena_alloc(clb_arena *a, size_t size, size_t align);
CLB_API void       clb_arena_reset(clb_arena *a);
CLB_API size_t     clb_arena_used(const clb_arena *a);
CLB_API size_t     clb_arena_capacity(const clb_arena *a);

CLB_EXTERN_C_END
#endif /* CLB_ALLOC_H */