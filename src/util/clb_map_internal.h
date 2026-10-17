#ifndef CLB_MAP_INTERNAL_H
#define CLB_MAP_INTERNAL_H
#include <stddef.h>
#include <stdint.h>
#include "colibri/clb_error.h"

typedef struct clb_map_slot {
    char    *key;
    void    *value;
    uint64_t hash;
} clb_map_slot;

typedef struct clb_map {
    clb_map_slot *slots;
    size_t        cap;
    size_t        len;
} clb_map;

clb_status clb_map_init(clb_map *m, size_t initial_cap);
void       clb_map_free(clb_map *m);
clb_status clb_map_put(clb_map *m, const char *key, void *value);
void      *clb_map_get(const clb_map *m, const char *key);
size_t     clb_map_len(const clb_map *m);
#endif