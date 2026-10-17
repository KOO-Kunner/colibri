/* SPDX-License-Identifier: MIT */
#define _POSIX_C_SOURCE 200112L

#include "colibri/clb_alloc.h"
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

void *clb_malloc(size_t size)
{
    if (size == 0) size = 1;
    return malloc(size);
}

void *clb_calloc(size_t count, size_t size)
{
    size_t total;
    if (!clb_size_mul(count, size, &total)) return NULL;
    if (total == 0) total = 1;
    return calloc(1, total);
}

void *clb_realloc(void *ptr, size_t size)
{
    if (size == 0) size = 1;
    return realloc(ptr, size);
}

void clb_free(void *ptr) { free(ptr); }

void *clb_aligned_alloc(size_t alignment, size_t size)
{
    if (alignment == 0 || (alignment & (alignment - 1)) != 0) return NULL;
    if (size == 0) size = 1;
#if defined(_WIN32)
    return _aligned_malloc(size, alignment);
#else
    void *p = NULL;
    if (posix_memalign(&p, alignment, size) != 0) return NULL;
    return p;
#endif
}

void clb_aligned_free(void *ptr)
{
    if (!ptr) return;
#if defined(_WIN32)
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

bool clb_size_mul(size_t a, size_t b, size_t *out)
{
    if (a != 0 && b > (size_t)-1 / a) return false;
    *out = a * b;
    return true;
}

bool clb_size_add(size_t a, size_t b, size_t *out)
{
    if (b > (size_t)-1 - a) return false;
    *out = a + b;
    return true;
}

/* ---------------- Arena ---------------- */

struct clb_arena {
    uint8_t *base;
    size_t   capacity;
    size_t   offset;
};

clb_arena *clb_arena_create(size_t capacity)
{
    clb_arena *a = (clb_arena *)clb_calloc(1, sizeof(*a));
    if (!a) return NULL;
    a->base = (uint8_t *)clb_aligned_alloc(CLB_DEFAULT_ALIGN, capacity);
    if (!a->base) { clb_free(a); return NULL; }
    a->capacity = capacity;
    a->offset   = 0;
    return a;
}

void clb_arena_destroy(clb_arena *a)
{
    if (!a) return;
    clb_aligned_free(a->base);
    clb_free(a);
}

void *clb_arena_alloc(clb_arena *a, size_t size, size_t align)
{
    if (!a) return NULL;
    if (align == 0) align = 8;
    size_t aligned = (a->offset + align - 1) & ~(align - 1);
    if (aligned > a->capacity || size > a->capacity - aligned) return NULL;
    void *p = a->base + aligned;
    a->offset = aligned + size;
    memset(p, 0, size);
    return p;
}

void   clb_arena_reset(clb_arena *a)  { if (a) a->offset = 0; }
size_t clb_arena_used(const clb_arena *a) { return a ? a->offset : 0; }
size_t clb_arena_capacity(const clb_arena *a) { return a ? a->capacity : 0; }