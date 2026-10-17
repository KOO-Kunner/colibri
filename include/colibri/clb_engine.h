/* SPDX-License-Identifier: MIT */
#ifndef CLB_ENGINE_H
#define CLB_ENGINE_H

#include "clb_model.h"
#include "clb_context.h"
#include "clb_sampler.h"

CLB_EXTERN_C_BEGIN

typedef struct clb_engine clb_engine;

typedef struct clb_engine_params {
    const char *model_path;
    int32_t     n_ctx;
    int         n_threads;
    bool        use_mmap;
    clb_log_level log_level;
} clb_engine_params;

typedef struct clb_gen_params {
    const char *prompt;
    int32_t     n_predict;
    clb_sampler_params sampler;
    bool        echo_prompt;
} clb_gen_params;

typedef bool (*clb_stream_fn)(const char *piece, int32_t token, void *user);

CLB_API clb_engine_params clb_engine_default_params(void);
CLB_API clb_gen_params    clb_gen_default_params(void);

CLB_API clb_status clb_engine_create(const clb_engine_params *p, clb_engine **out);
CLB_API void       clb_engine_free(clb_engine *e);

CLB_API clb_status clb_engine_generate(clb_engine *e, const clb_gen_params *gp,
                                       clb_stream_fn cb, void *user,
                                       int32_t *out_n_generated);

CLB_API const clb_model   *clb_engine_model(const clb_engine *e);
CLB_API const clb_context *clb_engine_context(const clb_engine *e);

CLB_EXTERN_C_END
#endif /* CLB_ENGINE_H */