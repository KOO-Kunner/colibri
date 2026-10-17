/* SPDX-License-Identifier: MIT */
#ifndef CLB_CONTEXT_H
#define CLB_CONTEXT_H

#include "clb_model.h"

CLB_EXTERN_C_BEGIN

typedef struct clb_context clb_context;

typedef struct clb_context_params {
    int32_t n_ctx;
    int32_t n_batch;
    int     n_threads;
    bool    logits_all;
} clb_context_params;

CLB_API clb_context_params clb_context_default_params(void);

CLB_API clb_status clb_context_create(const clb_model *m,
                                      const clb_context_params *p,
                                      clb_context **out);
CLB_API void       clb_context_free(clb_context *ctx);
CLB_API void       clb_context_reset(clb_context *ctx);

CLB_API clb_status clb_context_eval(clb_context *ctx, int32_t token,
                                    float **out_logits);

CLB_API clb_status clb_context_eval_batch(clb_context *ctx,
                                          const int32_t *tokens, int32_t n,
                                          float **out_logits);

CLB_API int32_t clb_context_n_past(const clb_context *ctx);
CLB_API int32_t clb_context_n_ctx(const clb_context *ctx);
CLB_API clb_status clb_context_set_n_past(clb_context *ctx, int32_t n);

CLB_EXTERN_C_END
#endif /* CLB_CONTEXT_H */