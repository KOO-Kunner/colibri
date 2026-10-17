/* SPDX-License-Identifier: MIT */
#include "colibri/clb_engine.h"
#include "colibri/clb_log.h"
#include "colibri/clb_alloc.h"
#include "colibri/clb_cpu.h"
#include <string.h>
#include <stdlib.h>

struct clb_engine {
    clb_model   *model;
    clb_context *ctx;
    clb_engine_params p;
};

clb_engine_params clb_engine_default_params(void)
{
    clb_engine_params p;
    memset(&p, 0, sizeof(p));
    p.model_path = NULL;
    p.n_ctx      = 0;
    p.n_threads  = 0;
    p.use_mmap   = true;
    p.log_level  = CLB_LOG_WARN;
    return p;
}

clb_gen_params clb_gen_default_params(void)
{
    clb_gen_params g;
    memset(&g, 0, sizeof(g));
    g.prompt      = "";
    g.n_predict   = 128;
    g.echo_prompt = false;
    g.sampler     = clb_sampler_default_params();
    return g;
}

clb_status clb_engine_create(const clb_engine_params *params, clb_engine **out)
{
    if (!params || !params->model_path || !out) return CLB_ERR_INVALID_ARG;

    clb_log_set_level(params->log_level);

    clb_engine *e = (clb_engine *)clb_calloc(1, sizeof(*e));
    if (!e) return CLB_ERR_OUT_OF_MEMORY;
    e->p = *params;

    clb_model_params mp = clb_model_default_params();
    mp.use_mmap  = params->use_mmap;
    mp.n_threads = params->n_threads;
    mp.log_level = params->log_level;

    clb_status st = clb_model_load(params->model_path, &mp, &e->model);
    if (st != CLB_OK) { clb_free(e); return st; }

    clb_context_params cp = clb_context_default_params();
    cp.n_ctx     = params->n_ctx;
    cp.n_threads = params->n_threads;

    st = clb_context_create(e->model, &cp, &e->ctx);
    if (st != CLB_OK) { clb_model_free(e->model); clb_free(e); return st; }

    *out = e;
    return CLB_OK;
}

void clb_engine_free(clb_engine *e)
{
    if (!e) return;
    if (e->ctx)   clb_context_free(e->ctx);
    if (e->model) clb_model_free(e->model);
    clb_free(e);
}

const clb_model   *clb_engine_model(const clb_engine *e)
{
    return e ? e->model : NULL;
}
const clb_context *clb_engine_context(const clb_engine *e)
{
    return e ? e->ctx : NULL;
}

clb_status clb_engine_generate(clb_engine *e, const clb_gen_params *gp,
                               clb_stream_fn cb, void *user,
                               int32_t *out_n_generated)
{
    if (!e || !gp) return CLB_ERR_INVALID_ARG;

    clb_context_reset(e->ctx);

    const clb_tokenizer *tk = e->model->tokenizer;
    const int32_t n_vocab   = e->model->hp.n_vocab;
    const int32_t n_ctx     = clb_context_n_ctx(e->ctx);

    /* 1) prompt → token ids */
    int32_t *tokens = (int32_t *)clb_malloc((size_t)n_ctx * sizeof(int32_t));
    if (!tokens) return CLB_ERR_OUT_OF_MEMORY;

    clb_status st;
    int32_t n_prompt = clb_tokenizer_encode(
        tk, gp->prompt ? gp->prompt : "", tokens, n_ctx, true, false, &st);
    if (st != CLB_OK || n_prompt == 0) { clb_free(tokens); return CLB_ERR_TOKENIZE; }

    if (gp->echo_prompt && cb) {
        for (int32_t i = 0; i < n_prompt; i++) {
            char piece[256];
            clb_tokenizer_decode_token(tk, tokens[i], piece, sizeof(piece));
            if (!cb(piece, tokens[i], user)) { clb_free(tokens); return CLB_OK; }
        }
    }

    /* 2) 预填充 */
    float *logits = NULL;
    st = clb_context_eval_batch(e->ctx, tokens, n_prompt, &logits);
    clb_free(tokens);
    if (st != CLB_OK) return st;

    /* 3) 采样循环 */
    clb_sampler_params sp = gp->sampler;
    sp.n_vocab = n_vocab;
    clb_sampler *sampler = clb_sampler_create(&sp, &st);
    if (!sampler) return st;

    int32_t n_gen = 0;
    int32_t n_predict = gp->n_predict;
    if (n_predict < 0) n_predict = n_ctx - clb_context_n_past(e->ctx);

    for (int32_t i = 0; i < n_predict; i++) {
        if (clb_context_n_past(e->ctx) >= n_ctx) break;

        int32_t next = clb_sampler_sample(sampler, logits);
        clb_sampler_accept(sampler, next);

        if (clb_tokenizer_is_eog(tk, next)) break;

        char piece[256];
        clb_tokenizer_decode_token(tk, next, piece, sizeof(piece));
        if (cb && !cb(piece, next, user)) { n_gen++; break; }

        st = clb_context_eval(e->ctx, next, &logits);
        if (st != CLB_OK) { n_gen = -1; break; }
        n_gen++;
    }

    clb_sampler_free(sampler);
    if (out_n_generated) *out_n_generated = n_gen;
    return st;
}