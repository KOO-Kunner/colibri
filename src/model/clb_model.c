/* SPDX-License-Identifier: MIT */
#include "colibri/clb_model.h"
#include "colibri/clb_log.h"
#include "colibri/clb_alloc.h"
#include "colibri/clb_cpu.h"
#include "colibri/clb_quant.h"
#include "clb_map_internal.h"
#include <string.h>
#include <stdio.h>

#define CLB_MAGIC        "COLIBRI"
#define CLB_FORMAT_VERSION 1u
#define CLB_HDR_SIZE     128u

/* 外部 mmap 接口（来自 src/util/clb_io.c） */
extern void  *clb__io_mmap_open(const char *path, size_t *size,
                                clb_status *st, void **handle);
extern void   clb__io_mmap_close(void *handle);
extern clb_status clb__io_read_all(const char *path, uint8_t **out, size_t *out_size);

typedef struct clb_model_priv {
    uint8_t       *base;
    size_t         size;
    bool           is_mmap;
    void          *map_handle;
    clb_map        tensor_index;
    clb_tensor    *tensor_pool;
    int32_t        n_tensors;
} clb_model_priv;

/* ---- 小端读取辅助 ---- */
static uint32_t rd_u32(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8)
         | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}
static uint64_t rd_u64(const uint8_t *p) {
    return (uint64_t)rd_u32(p) | ((uint64_t)rd_u32(p + 4) << 32);
}
static int64_t rd_i64(const uint8_t *p) { return (int64_t)rd_u64(p); }
static float rd_f32(const uint8_t *p) {
    uint32_t v = rd_u32(p);
    float f;
    memcpy(&f, &v, 4);
    return f;
}
static uint64_t align_up_u64(uint64_t v, uint64_t a) {
    return (v + a - 1) & ~(a - 1);
}

clb_model_params clb_model_default_params(void)
{
    clb_model_params p;
    memset(&p, 0, sizeof(p));
    p.use_mmap    = true;
    p.use_mlock   = false;
    p.n_threads   = 0;
    p.log_level   = CLB_LOG_WARN;
    return p;
}

/* 把 "blk.0.attn_q.weight" 规范化为 "blk_0_attn_q_weight" */
static void normalize_name(char *dst, size_t dst_size, const char *src)
{
    size_t j = 0;
    for (size_t i = 0; src[i] && j + 1 < dst_size; i++) {
        char c = src[i];
        if (c == '.') c = '_';
        dst[j++] = c;
    }
    dst[j] = '\0';
}

static void bind_layer(clb_model *m, clb_model_priv *pv, int i,
                       const char *suffix, clb_tensor *out)
{
    char key[CLB_TENSOR_NAME_MAX];
    snprintf(key, sizeof(key), "blk_%d_%s", i, suffix);
    clb_tensor *t = (clb_tensor *)clb_map_get(&pv->tensor_index, key);
    if (t) *out = *t;
    else   memset(out, 0, sizeof(*out));
    (void)m;
}

clb_status clb_model_load(const char *path, const clb_model_params *params,
                          clb_model **out_model)
{
    if (!path || !out_model) return CLB_ERR_INVALID_ARG;
    clb_model_params def = clb_model_default_params();
    if (!params) params = &def;

    clb_model      *m  = (clb_model *)clb_calloc(1, sizeof(*m));
    clb_model_priv *pv = (clb_model_priv *)clb_calloc(1, sizeof(*pv));
    if (!m || !pv) { clb_free(m); clb_free(pv); return CLB_ERR_OUT_OF_MEMORY; }
    m->_priv = pv;

    clb_status st = CLB_OK;
    if (params->use_mmap) {
        pv->base = (uint8_t *)clb__io_mmap_open(path, &pv->size, &st, &pv->map_handle);
        pv->is_mmap = true;
    } else {
        st = clb__io_read_all(path, &pv->base, &pv->size);
        pv->is_mmap = false;
    }
    if (!pv->base) {
        CLB_LOGE("failed to open model: %s", path);
        goto fail;
    }
    if (pv->size < CLB_HDR_SIZE) {
        CLB_LOGE("file too small");
        st = CLB_ERR_FORMAT;
        goto fail;
    }

    const uint8_t *h = pv->base;
    if (memcmp(h, CLB_MAGIC, 7) != 0) {
        CLB_LOGE("bad magic (expected %s)", CLB_MAGIC);
        st = CLB_ERR_FORMAT;
        goto fail;
    }

    uint32_t ver = rd_u32(h + 8);
    if (ver != CLB_FORMAT_VERSION) {
        CLB_LOGE("unsupported version %u", ver);
        st = CLB_ERR_FORMAT;
        goto fail;
    }

    m->hp.n_vocab       = (int32_t)rd_u32(h + 16);
    m->hp.n_embd        = (int32_t)rd_u32(h + 20);
    m->hp.n_layer       = (int32_t)rd_u32(h + 24);
    m->hp.n_head        = (int32_t)rd_u32(h + 28);
    m->hp.n_head_kv     = (int32_t)rd_u32(h + 32);
    m->hp.n_ff          = (int32_t)rd_u32(h + 36);
    m->hp.n_ctx_train   = (int32_t)rd_u32(h + 40);
    uint32_t n_tensors  = rd_u32(h + 44);
    m->hp.rope_theta    = rd_f32(h + 48);
    m->hp.rms_eps       = rd_f32(h + 52);
    m->hp.tie_embeddings = rd_u32(h + 56) != 0;

    if (m->hp.n_head_kv <= 0) m->hp.n_head_kv = m->hp.n_head;
    m->head_dim  = m->hp.n_embd / m->hp.n_head;
    m->n_embd_kv = m->hp.n_head_kv * m->head_dim;

    CLB_LOGI("model: vocab=%d embd=%d layer=%d head=%d/%d ff=%d ctx=%d",
             m->hp.n_vocab, m->hp.n_embd, m->hp.n_layer,
             m->hp.n_head, m->hp.n_head_kv, m->hp.n_ff, m->hp.n_ctx_train);

    /* ---- Tokenizer 区 ---- */
    uint64_t off = CLB_HDR_SIZE;

    uint32_t tok_model_len = rd_u32(pv->base + off); off += 4;
    if (off + tok_model_len > pv->size) { st = CLB_ERR_FORMAT; goto fail; }
    off += tok_model_len;

    uint32_t n_tokens = rd_u32(pv->base + off); off += 4;
    char    **tok_strings = (char **)clb_calloc(n_tokens, sizeof(char *));
    uint32_t *tok_lens    = (uint32_t *)clb_calloc(n_tokens, sizeof(uint32_t));
    if (!tok_strings || !tok_lens) {
        clb_free(tok_strings); clb_free(tok_lens);
        st = CLB_ERR_OUT_OF_MEMORY; goto fail;
    }
    for (uint32_t i = 0; i < n_tokens; i++) {
        uint32_t len = rd_u32(pv->base + off); off += 4;
        if (off + len > pv->size) { clb_free(tok_strings); clb_free(tok_lens); st = CLB_ERR_FORMAT; goto fail; }
        tok_strings[i] = (char *)(pv->base + off);
        tok_lens[i]    = len;
        off += len;
    }

    uint32_t n_merges = rd_u32(pv->base + off); off += 4;
    char **merges = (char **)clb_calloc(n_merges ? n_merges : 1, sizeof(char *));
    if (!merges) {
        clb_free(tok_strings); clb_free(tok_lens);
        st = CLB_ERR_OUT_OF_MEMORY; goto fail;
    }
    for (uint32_t i = 0; i < n_merges; i++) {
        uint32_t len = rd_u32(pv->base + off); off += 4;
        if (off + len > pv->size) { clb_free(tok_strings); clb_free(tok_lens); clb_free(merges); st = CLB_ERR_FORMAT; goto fail; }
        merges[i] = (char *)(pv->base + off);
        off += len;
    }

    /* ---- 张量表 ---- */
    pv->n_tensors   = (int32_t)n_tensors;
    pv->tensor_pool = (clb_tensor *)clb_calloc(n_tensors ? n_tensors : 1,
                                               sizeof(clb_tensor));
    if (!pv->tensor_pool) {
        clb_free(tok_strings); clb_free(tok_lens); clb_free(merges);
        st = CLB_ERR_OUT_OF_MEMORY; goto fail;
    }
    clb_map_init(&pv->tensor_index, (size_t)n_tensors * 2);

    for (uint32_t i = 0; i < n_tensors; i++) {
        uint32_t name_len = rd_u32(pv->base + off); off += 4;
        if (name_len == 0 || name_len >= CLB_TENSOR_NAME_MAX) {
            st = CLB_ERR_FORMAT; goto fail;
        }
        char raw_name[CLB_TENSOR_NAME_MAX];
        memcpy(raw_name, pv->base + off, name_len);
        raw_name[name_len] = '\0';
        off += name_len;

        uint8_t  dtype    = pv->base[off]; off += 1;
        uint8_t  n_dims   = pv->base[off]; off += 1;
        off += 2;  /* pad */
        uint64_t data_off = rd_u64(pv->base + off); off += 8;
        int64_t  ne[CLB_MAX_DIMS] = {1, 1, 1, 1};
        for (int d = 0; d < 4; d++) { ne[d] = rd_i64(pv->base + off); off += 8; }

        char norm[CLB_TENSOR_NAME_MAX];
        normalize_name(norm, sizeof(norm), raw_name);

        clb_tensor *t = &pv->tensor_pool[i];
        clb_status s2 = clb_tensor_init(t, norm, (clb_dtype)dtype, n_dims, ne);
        if (s2 != CLB_OK) { st = s2; goto fail; }

        /* 暂时存相对偏移，稍后修正 */
        t->data  = (void *)(uintptr_t)data_off;
        t->owned = false;
        clb_map_put(&pv->tensor_index, norm, t);
    }

    uint64_t data_start = align_up_u64(off, 64);
    if (data_start > pv->size) { st = CLB_ERR_FORMAT; goto fail; }

    /* 修正数据指针 */
    for (uint32_t i = 0; i < n_tensors; i++) {
        clb_tensor *t = &pv->tensor_pool[i];
        uint64_t rel  = (uint64_t)(uintptr_t)t->data;
        t->data = pv->base + data_start + rel;
    }

    /* ---- 绑定到 clb_model 字段 ---- */
    clb_tensor *te = (clb_tensor *)clb_map_get(&pv->tensor_index, "token_embd_weight");
    if (te) m->token_embd = *te;

    clb_tensor *on = (clb_tensor *)clb_map_get(&pv->tensor_index, "output_norm_weight");
    if (on) m->output_norm = *on;

    clb_tensor *ow = (clb_tensor *)clb_map_get(&pv->tensor_index, "output_weight");
    if (ow) m->output = *ow;

    m->layers = (clb_layer *)clb_calloc((size_t)m->hp.n_layer, sizeof(clb_layer));
    if (!m->layers) { st = CLB_ERR_OUT_OF_MEMORY; goto fail; }

    for (int i = 0; i < m->hp.n_layer; i++) {
        clb_layer *L = &m->layers[i];
        bind_layer(m, pv, i, "attn_norm_weight",    &L->attn_norm);
        bind_layer(m, pv, i, "attn_q_weight",       &L->wq);
        bind_layer(m, pv, i, "attn_k_weight",       &L->wk);
        bind_layer(m, pv, i, "attn_v_weight",       &L->wv);
        bind_layer(m, pv, i, "attn_output_weight",  &L->wo);
        bind_layer(m, pv, i, "ffn_norm_weight",     &L->ffn_norm);
        bind_layer(m, pv, i, "ffn_gate_weight",     &L->w_gate);
        bind_layer(m, pv, i, "ffn_up_weight",       &L->w_up);
        bind_layer(m, pv, i, "ffn_down_weight",     &L->w_down);

        if (!L->wq.data || !L->wo.data) {
            CLB_LOGE("layer %d missing weights (expected blk_%d_*)", i, i);
            st = CLB_ERR_FORMAT;
            goto fail;
        }
    }

    /* ---- Tokenizer ---- */
    clb_tokenizer_params tp;
    memset(&tp, 0, sizeof(tp));
    tp.tokens     = (const char *const *)tok_strings;
    tp.token_lens = tok_lens;
    tp.n_vocab    = (int32_t)n_tokens;
    tp.merges     = (const char *const *)merges;
    tp.n_merges   = n_merges;
    tp.bos_id     = 1;
    tp.eos_id     = 2;
    tp.unk_id     = 0;
    m->tokenizer  = clb_tokenizer_create(&tp, &st);
    if (!m->tokenizer) {
        clb_free(tok_strings); clb_free(tok_lens); clb_free(merges);
        goto fail;
    }

    clb_free(tok_strings);
    clb_free(tok_lens);
    clb_free(merges);

    *out_model = m;
    return CLB_OK;

fail:
    clb_model_free(m);
    return st != CLB_OK ? st : CLB_ERR_FORMAT;
}

void clb_model_free(clb_model *m)
{
    if (!m) return;
    clb_model_priv *pv = (clb_model_priv *)m->_priv;
    if (m->tokenizer) clb_tokenizer_free(m->tokenizer);
    if (pv) {
        clb_map_free(&pv->tensor_index);
        clb_free(pv->tensor_pool);
        if (pv->base) {
            if (pv->is_mmap) clb__io_mmap_close(pv->map_handle);
            else             clb_free(pv->base);
        }
        clb_free(pv);
    }
    clb_free(m->layers);
    clb_free(m);
}

clb_tensor *clb_model_find_tensor(clb_model *m, const char *name)
{
    if (!m || !m->_priv) return NULL;
    clb_model_priv *pv = (clb_model_priv *)m->_priv;
    char norm[CLB_TENSOR_NAME_MAX];
    normalize_name(norm, sizeof(norm), name);
    return (clb_tensor *)clb_map_get(&pv->tensor_index, norm);
}

size_t clb_model_nbytes(const clb_model *m)
{
    if (!m || !m->_priv) return 0;
    const clb_model_priv *pv = (const clb_model_priv *)m->_priv;
    return pv->size;
}