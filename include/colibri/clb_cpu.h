/* SPDX-License-Identifier: MIT */
#ifndef CLB_CPU_H
#define CLB_CPU_H

#include <stdbool.h>
#include <stddef.h>
#include "clb_export.h"

CLB_EXTERN_C_BEGIN

typedef struct clb_cpu_features {
    bool sse2;
    bool sse41;
    bool avx;
    bool avx2;
    bool avx512f;
    bool fma;
    bool neon;
    int  n_physical_cores;
    int  n_logical_cores;
    int  cache_line_size;
} clb_cpu_features;

CLB_API const clb_cpu_features *clb_cpu_detect(void);
CLB_API int  clb_cpu_default_threads(void);
CLB_API void clb_cpu_summary(char *buf, size_t buf_size);

CLB_EXTERN_C_END
#endif /* CLB_CPU_H */