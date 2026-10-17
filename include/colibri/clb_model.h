/* SPDX-License-Identifier: MIT */
#ifndef CLB_MODEL_H
#define CLB_MODEL_H

#include <stdint.h>
#include "clb_tensor.h"
#include "clb_tokenizer.h"
#include "clb_log.h" 

CLB_EXTERN_C_BEGIN

#define CLB_MAX_LAYERS 256

typedef struct clb_hparams {
    int32_t n_vocab;
    int32_t n_embd;
    int32_t n_layer;
    int32_t n_head;
    int32_t n_head_kv;
    int32_t n_ff;
    int32_t n_ctx_train;
    float   rope_theta;
    float   rms_eps;
    bool    tie_embeddings;
} clb_hparams;

typedef struct clb_layer {
    clb_tensor attn_norm;
    clb_tensor wq;
    clb_tensor wk;
    clb_tensor wv;
    clb_tensor wo;
    clb_tensor ffn_norm;
    clb_tensor w_gate;
    clb_tensor w_up;
    clb_tensor w_down;
} clb_layer;

typedef struct clb_model {
    clb_hparams    hp;
    int32_t        head_dim;
    int32_t        n_embd_kv;
    clb_tensor     token_embd;
    clb_tensor     output_norm;
    clb_tensor     output;
    clb_layer     *layers;
    clb_tokenizer *tokenizer;
    void          *_priv;
} clb_model;

typedef struct clb_model_params {
    bool use_mmap;
    bool use_mlock;
    int  n_threads;
    clb_log_level log_level;
} clb_model_params;

CLB_API clb_model_params clb_model_default_params(void);

CLB_API clb_status clb_model_load(const char *path, const clb_model_params *p,
                                  clb_model **out);
CLB_API void       clb_model_free(clb_model *m);
CLB_API clb_tensor *clb_model_find_tensor(clb_model *m, const char *name);
CLB_API size_t     clb_model_nbytes(const clb_model *m);

CLB_EXTERN_C_END
#endif /* CLB_MODEL_H */