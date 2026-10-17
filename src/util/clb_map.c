/* SPDX-License-Identifier: MIT */
#include "clb_map_internal.h"
#include "colibri/clb_alloc.h"
#include <string.h>

static uint64_t fnv1a(const char *s)
{
    uint64_t h = 1469598103934665603ull;
    while (*s) { h ^= (uint8_t)*s++; h *= 1099511628211ull; }
    return h;
}

clb_status clb_map_init(clb_map *m, size_t initial_cap)
{
    if (initial_cap < 16) initial_cap = 16;
    size_t cap = 16;
    while (cap < initial_cap) cap <<= 1;

    m->slots = (clb_map_slot *)clb_calloc(cap, sizeof(clb_map_slot));
    if (!m->slots) return CLB_ERR_OUT_OF_MEMORY;
    m->cap = cap;
    m->len = 0;
    return CLB_OK;
}

void clb_map_free(clb_map *m)
{
    if (!m) return;
    for (size_t i = 0; i < m->cap; i++) clb_free(m->slots[i].key);
    clb_free(m->slots);
    m->slots = NULL; m->cap = 0; m->len = 0;
}

static clb_status map_grow(clb_map *m)
{
    size_t ncap = m->cap * 2;
    clb_map_slot *ns = (clb_map_slot *)clb_calloc(ncap, sizeof(clb_map_slot));
    if (!ns) return CLB_ERR_OUT_OF_MEMORY;
    for (size_t i = 0; i < m->cap; i++) {
        if (!m->slots[i].key) continue;
        size_t j = m->slots[i].hash & (ncap - 1);
        while (ns[j].key) j = (j + 1) & (ncap - 1);
        ns[j] = m->slots[i];
    }
    clb_free(m->slots);
    m->slots = ns;
    m->cap = ncap;
    return CLB_OK;
}

clb_status clb_map_put(clb_map *m, const char *key, void *value)
{
    if (m->len * 4 >= m->cap * 3) {
        clb_status st = map_grow(m);
        if (st != CLB_OK) return st;
    }
    uint64_t h = fnv1a(key);
    size_t i = h & (m->cap - 1);
    while (m->slots[i].key) {
        if (m->slots[i].hash == h && strcmp(m->slots[i].key, key) == 0) {
            m->slots[i].value = value; return CLB_OK;
        }
        i = (i + 1) & (m->cap - 1);
    }
    size_t klen = strlen(key) + 1;
    m->slots[i].key = (char *)clb_malloc(klen);
    if (!m->slots[i].key) return CLB_ERR_OUT_OF_MEMORY;
    memcpy(m->slots[i].key, key, klen);
    m->slots[i].hash  = h;
    m->slots[i].value = value;
    m->len++;
    return CLB_OK;
}

void *clb_map_get(const clb_map *m, const char *key)
{
    uint64_t h = fnv1a(key);
    size_t i = h & (m->cap - 1);
    while (m->slots[i].key) {
        if (m->slots[i].hash == h && strcmp(m->slots[i].key, key) == 0)
            return m->slots[i].value;
        i = (i + 1) & (m->cap - 1);
    }
    return NULL;
}

size_t clb_map_len(const clb_map *m) { return m->len; }
