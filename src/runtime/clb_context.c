/* SPDX-License-Identifier: MIT */
#include "colibri/clb_context.h"
#include "colibri/clb_ops.h"
#include "colibri/clb_log.h"
#include "colibri/clb_alloc.h"
#include "colibri/clb_cpu.h"
#include <math.h>
#include <string.h>

struct clb_context {
    const clb_model *model;
    clb_context_params p;
    clb_threadpool  *tp;

    int32_t n_ctx;
    int32_t n_past;

    /* KV cache: 每层 [n_ctx, n_embd_kv] f32 */
    float **kv_k;
    float **kv_v;

    /* 工作缓冲（一维 f32 张量） */
    clb_tensor t_x;
    clb_tensor t_xb;
    clb_tensor t_q;
    clb_tensor t_k;
    clb_tensor t_v;
    clb_tensor t_attn_out;
    clb_tensor t_ffn_gate;
    clb_tensor t_ffn_up;
    clb_tensor t_ffn_down;
    clb_tensor t_attn;      /* [n_head * n_ctx] scores */

    float *logits;
};

clb_context_params clb_context_default_params(void)
{
    clb_context_params p;
    memset(&p, 0, sizeof(p));
    p.n_ctx      = 0;
    p.n_batch    = 1;
    p.n_threads  = 0;
    p.logits_all = false;
    return p;
}

static clb_status alloc_1d(clb_tensor *t, const char *name, int64_t n)
{
    int64_t ne[1] = { n };
    return clb_tensor_alloc(t, name, CLB_DTYPE_F32, 1, ne);
}

clb_status clb_context_create(const clb_model *m, const clb_context_params *params,
                              clb_context **out)
{
    if (!m || !out) return CLB_ERR_INVALID_ARG;
    clb_context_params def = clb_context_default_params();
    if (!params) params = &def;

    clb_context *c = (clb_context *)clb_calloc(1, sizeof(*c));
    if (!c) return CLB_ERR_OUT_OF_MEMORY;

    c->model = m;
    c->p     = *params;
    c->n_ctx = params->n_ctx > 0 ? params->n_ctx : m->hp.n_ctx_train;
    if (c->n_ctx > m->hp.n_ctx_train) c->n_ctx = m->hp.n_ctx_train;
    c->n_past = 0;

    int nt = params->n_threads > 0 ? params->n_threads : clb_cpu_default_threads();
    clb_status st;
    c->tp = clb_threadpool_create(nt, &st);
    if (!c->tp) { clb_free(c); return st; }

    /* KV Cache */
    c->kv_k = (float **)clb_calloc((size_t)m->hp.n_layer, sizeof(float *));
    c->kv_v = (float **)clb_calloc((size_t)m->hp.n_layer, sizeof(float *));
    if (!c->kv_k || !c->kv_v) { clb_context_free(c); return CLB_ERR_OUT_OF_MEMORY; }

    size_t kv_elems = (size_t)c->n_ctx * (size_t)m->n_embd_kv;
    for (int i = 0; i < m->hp.n_layer; i++) {
        c->kv_k[i] = (float *)clb_aligned_alloc(CLB_DEFAULT_ALIGN,
                                                kv_elems * sizeof(float));
        c->kv_v[i] = (float *)clb_aligned_alloc(CLB_DEFAULT_ALIGN,
                                                kv_elems * sizeof(float));
        if (!c->kv_k[i] || !c->kv_v[i]) { clb_context_free(c); return CLB_ERR_OUT_OF_MEMORY; }
        memset(c->kv_k[i], 0, kv_elems * sizeof(float));
        memset(c->kv_v[i], 0, kv_elems * sizeof(float));
    }

    const int64_t n_embd    = m->hp.n_embd;
    const int64_t n_ff      = m->hp.n_ff;
    const int64_t n_q       = (int64_t)m->hp.n_head * m->head_dim;
    const int64_t n_kv      = m->n_embd_kv;
    const int64_t n_attn_sz = (int64_t)m->hp.n_head * c->n_ctx;

    if ((st = alloc_1d(&c->t_x,         "ctx_x",         n_embd)))    goto fail;
    if ((st = alloc_1d(&c->t_xb,        "ctx_xb",        n_embd)))    goto fail;
    if ((st = alloc_1d(&c->t_q,         "ctx_q",         n_q)))       goto fail;
    if ((st = alloc_1d(&c->t_k,         "ctx_k",         n_kv)))      goto fail;
    if ((st = alloc_1d(&c->t_v,         "ctx_v",         n_kv)))      goto fail;
    if ((st = alloc_1d(&c->t_attn_out,  "ctx_attn_out",  n_q)))       goto fail;
    if ((st = alloc_1d(&c->t_ffn_gate,  "ctx_ffn_gate",  n_ff)))      goto fail;
    if ((st = alloc_1d(&c->t_ffn_up,    "ctx_ffn_up",    n_ff)))      goto fail;
    if ((st = alloc_1d(&c->t_ffn_down,  "ctx_ffn_down",  n_embd)))    goto fail;
    if ((st = alloc_1d(&c->t_attn,      "ctx_attn",      n_attn_sz))) goto fail;

    c->logits = (float *)clb_aligned_alloc(CLB_DEFAULT_ALIGN,
                                           (size_t)m->hp.n_vocab * sizeof(float));
    if (!c->logits) { st = CLB_ERR_OUT_OF_MEMORY; goto fail; }

    *out = c;
    return CLB_OK;

fail:
    clb_context_free(c);
    return st;
}

void clb_context_free(clb_context *c)
{
    if (!c) return;

    if (c->kv_k) {
        for (int i = 0; i < c->model->hp.n_layer; i++) clb_aligned_free(c->kv_k[i]);
        clb_free(c->kv_k);
    }
    if (c->kv_v) {
        for (int i = 0; i < c->model->hp.n_layer; i++) clb_aligned_free(c->kv_v[i]);
        clb_free(c->kv_v);
    }

    clb_tensor_free(&c->t_x);
    clb_tensor_free(&c->t_xb);
    clb_tensor_free(&c->t_q);
    clb_tensor_free(&c->t_k);
    clb_tensor_free(&c->t_v);
    clb_tensor_free(&c->t_attn_out);
    clb_tensor_free(&c->t_ffn_gate);
    clb_tensor_free(&c->t_ffn_up);
    clb_tensor_free(&c->t_ffn_down);
    clb_tensor_free(&c->t_attn);

    clb_aligned_free(c->logits);
    if (c->tp) clb_threadpool_destroy(c->tp);
    clb_free(c);
}

void clb_context_reset(clb_context *c)
{
    if (!c) return;
    c->n_past = 0;
    size_t kv_elems = (size_t)c->n_ctx * (size_t)c->model->n_embd_kv;
    for (int i = 0; i < c->model->hp.n_layer; i++) {
        memset(c->kv_k[i], 0, kv_elems * sizeof(float));
        memset(c->kv_v[i], 0, kv_elems * sizeof(float));
    }
}

int32_t clb_context_n_past(const clb_context *c) { return c->n_past; }
int32_t clb_context_n_ctx(const clb_context *c)  { return c->n_ctx; }

clb_status clb_context_set_n_past(clb_context *c, int32_t n)
{
    if (!c || n < 0 || n > c->n_ctx) return CLB_ERR_INVALID_ARG;
    c->n_past = n;
    return CLB_OK;
}

/* ---- 单层前向 ---- */
static clb_status eval_layer(clb_context *c, int layer, int pos)
{
    const clb_model *m  = c->model;
    const clb_layer *L  = &m->layers[layer];

    const int n_embd    = m->hp.n_embd;
    const int head_dim  = m->head_dim;
    const int n_head    = m->hp.n_head;
    const int n_head_kv = m->hp.n_head_kv;
    const int n_embd_kv = m->n_embd_kv;
    const int n_ff      = m->hp.n_ff;
    const int kv_group  = n_head / n_head_kv;

    float *x          = (float *)c->t_x.data;
    float *xb         = (float *)c->t_xb.data;
    float *q          = (float *)c->t_q.data;
    float *k          = (float *)c->t_k.data;
    float *v          = (float *)c->t_v.data;
    float *attn_out   = (float *)c->t_attn_out.data;
    float *ffn_gate   = (float *)c->t_ffn_gate.data;
    float *ffn_up     = (float *)c->t_ffn_up.data;
    float *ffn_down   = (float *)c->t_ffn_down.data;
    float *scores_all = (float *)c->t_attn.data;

    clb_status st;

    /* 1) RMSNorm */
    clb_rmsnorm(xb, x, (const float *)L->attn_norm.data, n_embd, m->hp.rms_eps);

    /* 2) QKV 投影（源是 1 行，m=1） */
    {
        int64_t ne_src[1] = { n_embd };
        clb_tensor tsrc;
        clb_tensor_init(&tsrc, "src", CLB_DTYPE_F32, 1, ne_src);
        tsrc.data = xb;

        if ((st = clb_matmul(&c->t_q, &L->wq, &tsrc, c->tp))) return st;
        if ((st = clb_matmul(&c->t_k, &L->wk, &tsrc, c->tp))) return st;
        if ((st = clb_matmul(&c->t_v, &L->wv, &tsrc, c->tp))) return st;
    }

    /* 3) RoPE（对 Q 和 K） */
    clb_rope_apply(q, (int64_t)n_head * head_dim, n_head,
                   pos, m->hp.rope_theta, 1.0f);
    clb_rope_apply(k, (int64_t)n_head_kv * head_dim, n_head_kv,
                   pos, m->hp.rope_theta, 1.0f);

    /* 4) 写入 KV cache（该层第 pos 个位置） */
    float *k_cache = c->kv_k[layer] + (size_t)pos * n_embd_kv;
    float *v_cache = c->kv_v[layer] + (size_t)pos * n_embd_kv;
    memcpy(k_cache, k, (size_t)n_embd_kv * sizeof(float));
    memcpy(v_cache, v, (size_t)n_embd_kv * sizeof(float));

    /* 5) 多头注意力 */
    const float scale = 1.0f / sqrtf((float)head_dim);

    for (int h = 0; h < n_head; h++) {
        const int h_kv = h / kv_group;

        const float *Qh  = q + (size_t)h * head_dim;
        float       *out = attn_out + (size_t)h * head_dim;
        float       *scores = scores_all + (size_t)h * c->n_ctx;

        /* 5a) Q · K^T / sqrt(d) */
        for (int t = 0; t <= pos; t++) {
            const float *Kt = c->kv_k[layer] + (size_t)t * n_embd_kv
                            + (size_t)h_kv * head_dim;
            float s = 0.0f;
            for (int d = 0; d < head_dim; d++) s += Qh[d] * Kt[d];
            scores[t] = s * scale;
        }

        /* 5b) softmax */
        clb_softmax_inplace(scores, pos + 1);

        /* 5c) 加权求和 V */
        memset(out, 0, (size_t)head_dim * sizeof(float));
        for (int t = 0; t <= pos; t++) {
            const float *Vt = c->kv_v[layer] + (size_t)t * n_embd_kv
                            + (size_t)h_kv * head_dim;
            const float w = scores[t];
            for (int d = 0; d < head_dim; d++) out[d] += w * Vt[d];
        }
    }

    /* 6) 输出投影 + 残差 */
    {
        int64_t ne_src[1] = { (int64_t)n_head * head_dim };
        clb_tensor tsrc;
        clb_tensor_init(&tsrc, "attn_out", CLB_DTYPE_F32, 1, ne_src);
        tsrc.data = attn_out;

        if ((st = clb_matmul(&c->t_xb, &L->wo, &tsrc, c->tp))) return st;
        clb_add_inplace(x, xb, n_embd);
    }

    /* 7) FFN */
    clb_rmsnorm(xb, x, (const float *)L->ffn_norm.data, n_embd, m->hp.rms_eps);

    {
        int64_t ne_src[1] = { n_embd };
        clb_tensor tsrc;
        clb_tensor_init(&tsrc, "ffn_in", CLB_DTYPE_F32, 1, ne_src);
        tsrc.data = xb;

        if ((st = clb_matmul(&c->t_ffn_gate, &L->w_gate, &tsrc, c->tp))) return st;
        if ((st = clb_matmul(&c->t_ffn_up,   &L->w_up,   &tsrc, c->tp))) return st;
    }

    clb_silu_inplace(ffn_gate, n_ff);
    clb_mul_inplace(ffn_gate, ffn_up, n_ff);

    {
        int64_t ne_src[1] = { n_ff };
        clb_tensor tsrc;
        clb_tensor_init(&tsrc, "ffn_mid", CLB_DTYPE_F32, 1, ne_src);
        tsrc.data = ffn_gate;

        if ((st = clb_matmul(&c->t_ffn_down, &L->w_down, &tsrc, c->tp))) return st;
        clb_add_inplace(x, ffn_down, n_embd);
    }

    return CLB_OK;
}

clb_status clb_context_eval(clb_context *c, int32_t token, float **out_logits)
{
    if (!c) return CLB_ERR_INVALID_ARG;
    if (c->n_past >= c->n_ctx) return CLB_ERR_CONTEXT_FULL;
    if (token < 0 || token >= c->model->hp.n_vocab) return CLB_ERR_INVALID_ARG;

    const clb_model *m = c->model;
    const int n_embd = m->hp.n_embd;
    const int pos    = c->n_past;

    float *x = (float *)c->t_x.data;

    /* 1) 词嵌入查表 */
    const float *emb = clb_tensor_row_f32_c(&m->token_embd, token);
    memcpy(x, emb, (size_t)n_embd * sizeof(float));

    /* 2) 逐层前向 */
    for (int l = 0; l < m->hp.n_layer; l++) {
        clb_status st = eval_layer(c, l, pos);
        if (st != CLB_OK) return st;
    }

    /* 3) 输出归一化 */
    float *xb = (float *)c->t_xb.data;
    clb_rmsnorm(xb, x, (const float *)m->output_norm.data, n_embd, m->hp.rms_eps);

    /* 4) 输出投影 */
    {
        int64_t ne_src[1] = { n_embd };
        clb_tensor tsrc;
        clb_tensor_init(&tsrc, "final", CLB_DTYPE_F32, 1, ne_src);
        tsrc.data = xb;

        int64_t ne_out[1] = { m->hp.n_vocab };
        clb_tensor tout;
        clb_tensor_init(&tout, "logits", CLB_DTYPE_F32, 1, ne_out);
        tout.data = c->logits;

        const clb_tensor *w = m->output.data ? &m->output : &m->token_embd;
        clb_status st = clb_matmul(&tout, w, &tsrc, c->tp);
        if (st != CLB_OK) return st;
    }

    c->n_past++;
    if (out_logits) *out_logits = c->logits;
    return CLB_OK;
}

clb_status clb_context_eval_batch(clb_context *c, const int32_t *tokens, int32_t n,
                                  float **out_logits)
{
    if (!c || !tokens || n <= 0) return CLB_ERR_INVALID_ARG;
    float *logits = NULL;
    for (int32_t i = 0; i < n; i++) {
        clb_status st = clb_context_eval(c, tokens[i], &logits);
        if (st != CLB_OK) return st;
    }
    if (out_logits) *out_logits = logits;
    return CLB_OK;
}