/* SPDX-License-Identifier: MIT */
#ifndef CLB_TOKENIZER_H
#define CLB_TOKENIZER_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include "clb_error.h"

CLB_EXTERN_C_BEGIN

typedef struct clb_tokenizer clb_tokenizer;

typedef struct clb_tokenizer_params {
    const char *const *tokens;
    const uint32_t    *token_lens;
    int32_t            n_vocab;
    const char *const *merges;
    uint32_t           n_merges;
    int32_t            bos_id;
    int32_t            eos_id;
    int32_t            unk_id;
} clb_tokenizer_params;

CLB_API clb_tokenizer *clb_tokenizer_create(const clb_tokenizer_params *p, clb_status *st);
CLB_API void           clb_tokenizer_free(clb_tokenizer *tk);

CLB_API int32_t clb_tokenizer_encode(const clb_tokenizer *tk, const char *text,
                                     int32_t *ids, int32_t ids_cap,
                                     bool add_bos, bool add_eos, clb_status *st);

CLB_API int32_t clb_tokenizer_decode_token(const clb_tokenizer *tk,
                                           int32_t id, char *buf, size_t buf_size);

CLB_API bool clb_tokenizer_is_eog(const clb_tokenizer *tk, int32_t id);

CLB_API int32_t clb_tokenizer_n_vocab(const clb_tokenizer *tk);
CLB_API int32_t clb_tokenizer_bos_id(const clb_tokenizer *tk);
CLB_API int32_t clb_tokenizer_eos_id(const clb_tokenizer *tk);

CLB_EXTERN_C_END
#endif /* CLB_TOKENIZER_H */