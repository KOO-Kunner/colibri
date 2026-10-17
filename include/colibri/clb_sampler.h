/* SPDX-License-Identifier: MIT */
#ifndef CLB_SAMPLER_H
#define CLB_SAMPLER_H

#include <stdint.h>
#include <stdbool.h>
#include "clb_error.h"

CLB_EXTERN_C_BEGIN

typedef enum clb_sampler_mode {
    CLB_SAMPLER_GREEDY = 0,
    CLB_SAMPLER_TEMP   = 1
} clb_sampler_mode;

typedef struct clb_sampler_params {
    clb_sampler_mode mode;
    uint64_t seed;
    float    temp;
    int32_t  top_k;
    float    top_p;
    float    min_p;
    float    repeat_penalty;
    int32_t  repeat_last_n;
    int32_t  n_vocab;
} clb_sampler_params;

typedef struct clb_sampler clb_sampler;

CLB_API clb_sampler_params clb_sampler_default_params(void);

CLB_API clb_sampler *clb_sampler_create(const clb_sampler_params *p, clb_status *st);
CLB_API void         clb_sampler_free(clb_sampler *s);
CLB_API void         clb_sampler_reset(clb_sampler *s);

CLB_API void clb_sampler_accept(clb_sampler *s, int32_t token);
CLB_API int32_t clb_sampler_sample(clb_sampler *s, float *logits);

CLB_EXTERN_C_END
#endif /* CLB_SAMPLER_H */