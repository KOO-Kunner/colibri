/* SPDX-License-Identifier: MIT */
#include "colibri/clb_tokenizer.h"
#include "colibri/clb_alloc.h"
#include "colibri/clb_log.h"
#include "clb_map_internal.h"
#include <string.h>
#include <stdlib.h>

struct clb_tokenizer {
    char    **tokens;       /* n_vocab 个词元字符串 */
    uint32_t *token_lens;
    int32_t   n_vocab;

    clb_map   token_to_id;  /* token string -> id (intptr_t) */

    clb_map   merge_rank;   /* "a b" -> rank */
    int32_t   n_merges;

    int32_t   bos_id, eos_id, unk_id;
};

static int32_t find_or_unk(const clb_tokenizer *tk, const char *s, size_t len)
{
    char tmp[256];
    if (len >= sizeof(tmp)) return tk->unk_id;
    memcpy(tmp, s, len);
    tmp[len] = '\0';
    void *v = clb_map_get(&tk->token_to_id, tmp);
    return v ? (int32_t)(intptr_t)v : tk->unk_id;
}

clb_tokenizer *clb_tokenizer_create(const clb_tokenizer_params *p, clb_status *st)
{
    if (!p || !p->tokens || p->n_vocab <= 0) {
        if (st) *st = CLB_ERR_INVALID_ARG;
        return NULL;
    }

    clb_tokenizer *tk = (clb_tokenizer *)clb_calloc(1, sizeof(*tk));
    if (!tk) { if (st) *st = CLB_ERR_OUT_OF_MEMORY; return NULL; }

    tk->n_vocab  = p->n_vocab;
    tk->bos_id   = p->bos_id;
    tk->eos_id   = p->eos_id;
    tk->unk_id   = p->unk_id;
    tk->n_merges = (int32_t)p->n_merges;

    tk->tokens     = (char **)clb_calloc((size_t)p->n_vocab, sizeof(char *));
    tk->token_lens = (uint32_t *)clb_calloc((size_t)p->n_vocab, sizeof(uint32_t));
    if (!tk->tokens || !tk->token_lens) goto fail;

    if (clb_map_init(&tk->token_to_id, (size_t)p->n_vocab * 2) != CLB_OK)
        goto fail;

    for (int32_t i = 0; i < p->n_vocab; i++) {
        size_t len = p->token_lens ? p->token_lens[i] : strlen(p->tokens[i]);
        tk->tokens[i] = (char *)clb_malloc(len + 1);
        if (!tk->tokens[i]) goto fail;
        memcpy(tk->tokens[i], p->tokens[i], len);
        tk->tokens[i][len] = '\0';
        tk->token_lens[i]  = (uint32_t)len;
        clb_map_put(&tk->token_to_id, tk->tokens[i], (void *)(intptr_t)i);
    }

    if (clb_map_init(&tk->merge_rank,
                     (size_t)(p->n_merges ? p->n_merges * 2 : 16)) != CLB_OK)
        goto fail;

    for (uint32_t i = 0; i < p->n_merges; i++)
        clb_map_put(&tk->merge_rank, p->merges[i], (void *)(intptr_t)(i + 1));

    if (st) *st = CLB_OK;
    return tk;

fail:
    clb_tokenizer_free(tk);
    if (st) *st = CLB_ERR_OUT_OF_MEMORY;
    return NULL;
}

void clb_tokenizer_free(clb_tokenizer *tk)
{
    if (!tk) return;
    if (tk->tokens) {
        for (int32_t i = 0; i < tk->n_vocab; i++) clb_free(tk->tokens[i]);
        clb_free(tk->tokens);
    }
    clb_free(tk->token_lens);
    clb_map_free(&tk->token_to_id);
    clb_map_free(&tk->merge_rank);
    clb_free(tk);
}

int32_t clb_tokenizer_n_vocab(const clb_tokenizer *tk) { return tk->n_vocab; }
int32_t clb_tokenizer_bos_id(const clb_tokenizer *tk)  { return tk->bos_id; }
int32_t clb_tokenizer_eos_id(const clb_tokenizer *tk)  { return tk->eos_id; }

bool clb_tokenizer_is_eog(const clb_tokenizer *tk, int32_t id)
{
    return id == tk->eos_id;
}

int32_t clb_tokenizer_decode_token(const clb_tokenizer *tk, int32_t id,
                                   char *buf, size_t buf_size)
{
    if (id < 0 || id >= tk->n_vocab || !buf || buf_size == 0) return 0;
    size_t len = tk->token_lens[id];
    if (len >= buf_size) len = buf_size - 1;
    memcpy(buf, tk->tokens[id], len);
    buf[len] = '\0';
    return (int32_t)len;
}

/*
 * 简化 BPE：
 *   1) 从当前位置开始，尝试 1..32 字节的最长匹配
 *   2) 未命中则按 UTF-8 字符边界步进，单字节作为 unk
 * 交互式 prompt 场景下足够使用。
 */
int32_t clb_tokenizer_encode(const clb_tokenizer *tk, const char *text,
                             int32_t *ids, int32_t ids_cap,
                             bool add_bos, bool add_eos, clb_status *st)
{
    if (!tk || !text || !ids || ids_cap <= 0) {
        if (st) *st = CLB_ERR_INVALID_ARG;
        return 0;
    }

    int32_t n = 0;
    if (add_bos && n < ids_cap) ids[n++] = tk->bos_id;

    const char *p = text;
    while (*p && n < ids_cap) {
        size_t remaining = strlen(p);
        int32_t best_id  = -1;
        size_t  best_len = 0;

        size_t max_try = remaining < 32 ? remaining : 32;
        for (size_t L = max_try; L >= 1; L--) {
            int32_t id = find_or_unk(tk, p, L);
            if (id != tk->unk_id) { best_id = id; best_len = L; break; }
        }

        if (best_id < 0) {
            best_id = tk->unk_id;
            unsigned char c = (unsigned char)*p;
            if      ((c & 0x80) == 0x00) best_len = 1;
            else if ((c & 0xE0) == 0xC0) best_len = 2;
            else if ((c & 0xF0) == 0xE0) best_len = 3;
            else if ((c & 0xF8) == 0xF0) best_len = 4;
            else                         best_len = 1;
            if (best_len > remaining) best_len = remaining;
        }

        ids[n++] = best_id;
        p += best_len;
    }

    if (add_eos && n < ids_cap) ids[n++] = tk->eos_id;

    if (st) *st = CLB_OK;
    return n;
}