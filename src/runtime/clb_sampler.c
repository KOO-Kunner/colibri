/* SPDX-License-Identifier: MIT */
#include "colibri/clb_sampler.h"
#include "colibri/clb_alloc.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

struct clb_sampler {
    clb_sampler_params p;
    uint64_t rng;
    int32_t *hist;
    int32_t  hist_cap;
    int32_t  hist_len;
    int32_t  hist_pos;
    int32_t *idx;
    float   *probs;
};

clb_sampler_params clb_sampler_default_params(void)
{
    clb_sampler_params p;
    memset(&p, 0, sizeof(p));
    p.mode           = CLB_SAMPLER_TEMP;
    p.seed           = 0x9E3779B97F4A7C15ull;
    p.temp           = 0.8f;
    p.top_k          = 40;
    p.top_p          = 0.95f;
    p.min_p          = 0.0f;
    p.repeat_penalty = 1.1f;
    p.repeat_last_n  = 64;
    p.n_vocab        = 0;
    return p;
}

clb_sampler *clb_sampler_create(const clb_sampler_params *p, clb_status *st)
{
    if (!p || p->n_vocab <= 0) {
        if (st) *st = CLB_ERR_INVALID_ARG;
        return NULL;
    }
    clb_sampler *s = (clb_sampler *)clb_calloc(1, sizeof(*s));
    if (!s) { if (st) *st = CLB_ERR_OUT_OF_MEMORY; return NULL; }

    s->p   = *p;
    s->rng = p->seed ? p->seed : 0x9E3779B97F4A7C15ull;

    s->hist_cap = p->repeat_last_n > 0 ? p->repeat_last_n : 256;
    s->hist     = (int32_t *)clb_calloc((size_t)s->hist_cap, sizeof(int32_t));
    s->idx      = (int32_t *)clb_malloc((size_t)p->n_vocab * sizeof(int32_t));
    s->probs    = (float *)clb_malloc((size_t)p->n_vocab * sizeof(float));

    if (!s->hist || !s->idx || !s->probs) {
        clb_sampler_free(s);
        if (st) *st = CLB_ERR_OUT_OF_MEMORY;
        return NULL;
    }
    for (int32_t i = 0; i < p->n_vocab; i++) s->idx[i] = i;

    if (st) *st = CLB_OK;
    return s;
}

void clb_sampler_free(clb_sampler *s)
{
    if (!s) return;
    clb_free(s->hist);
    clb_free(s->idx);
    clb_free(s->probs);
    clb_free(s);
}

void clb_sampler_reset(clb_sampler *s)
{
    if (!s) return;
    s->hist_len = 0;
    s->hist_pos = 0;
}

void clb_sampler_accept(clb_sampler *s, int32_t token)
{
    if (!s || token < 0) return;
    s->hist[s->hist_pos] = token;
    s->hist_pos = (s->hist_pos + 1) % s->hist_cap;
    if (s->hist_len < s->hist_cap) s->hist_len++;
}

static uint64_t xorshift64(uint64_t *state)
{
    uint64_t x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    return x;
}

static float rand_f32(clb_sampler *s)
{
    return (float)(xorshift64(&s->rng) >> 11) * (1.0f / 9007199254740992.0f);
}

static void apply_repeat_penalty(clb_sampler *s, float *logits)
{
    if (s->p.repeat_penalty <= 1.0f || s->hist_len == 0) return;
    for (int32_t i = 0; i < s->hist_len; i++) {
        int32_t t = s->hist[(s->hist_pos - 1 - i + s->hist_cap * 2) % s->hist_cap];
        if (t < 0 || t >= s->p.n_vocab) continue;
        float v = logits[t];
        logits[t] = v > 0.0f ? v / s->p.repeat_penalty : v * s->p.repeat_penalty;
    }
}

/*
 * 最小堆版本：排序完成后 idx 为「概率降序」。
 *   idx[0]   = 概率最高的 token
 *   idx[n-1] = 概率最低的 token
 */
static void sort_idx_by_prob(int32_t *idx, int32_t n, const float *probs)
{
    #define CLB_SWAP(a, b) do { int32_t _t = (a); (a) = (b); (b) = _t; } while (0)

    /* 构建最小堆 */
    for (int32_t i = n / 2 - 1; i >= 0; i--) {
        int32_t root = i;
        for (;;) {
            int32_t child = 2 * root + 1;
            if (child >= n) break;
            if (child + 1 < n && probs[idx[child + 1]] < probs[idx[child]]) child++;
            if (probs[idx[child]] < probs[idx[root]]) {
                CLB_SWAP(idx[root], idx[child]);
                root = child;
            } else break;
        }
    }

    /* 反复把根（最小）换到末尾 —— 结果降序 */
    for (int32_t end = n - 1; end > 0; end--) {
        CLB_SWAP(idx[0], idx[end]);
        int32_t root = 0;
        for (;;) {
            int32_t child = 2 * root + 1;
            if (child >= end) break;
            if (child + 1 < end && probs[idx[child + 1]] < probs[idx[child]]) child++;
            if (probs[idx[child]] < probs[idx[root]]) {
                CLB_SWAP(idx[root], idx[child]);
                root = child;
            } else break;
        }
    }

    #undef CLB_SWAP
}

int32_t clb_sampler_sample(clb_sampler *s, float *logits)
{
    const int32_t n = s->p.n_vocab;

    /* 1) 先应用重复惩罚（贪心和随机采样都需要） */
    apply_repeat_penalty(s, logits);

    /* 2) 贪心：直接 argmax */
    if (s->p.mode == CLB_SAMPLER_GREEDY) {
        int32_t best = 0;
        for (int32_t i = 1; i < n; i++)
            if (logits[i] > logits[best]) best = i;
        return best;
    }

    /* 3) softmax（带温度） */
    float maxv = logits[0];
    for (int32_t i = 1; i < n; i++) if (logits[i] > maxv) maxv = logits[i];

    const float inv_t = s->p.temp > 0.0f ? 1.0f / s->p.temp : 1.0f;
    float sum = 0.0f;
    for (int32_t i = 0; i < n; i++) {
        const float v = expf((logits[i] - maxv) * inv_t);
        s->probs[i] = v;
        sum += v;
    }
    if (sum <= 0.0f) sum = 1.0f;
    for (int32_t i = 0; i < n; i++) s->probs[i] /= sum;

    /* 4) 排序（概率降序） */
    for (int32_t i = 0; i < n; i++) s->idx[i] = i;
    sort_idx_by_prob(s->idx, n, s->probs);

    /* 5) top_k / top_p / min_p 截断 */
    int32_t keep = n;
    if (s->p.top_k > 0 && s->p.top_k < keep) keep = s->p.top_k;

    if (s->p.top_p < 1.0f || s->p.min_p > 0.0f) {
        float cum = 0.0f;
        int32_t k2 = 0;
        for (int32_t i = 0; i < keep; i++) {
            const float p = s->probs[s->idx[i]];
            if (s->p.min_p > 0.0f && p < s->p.min_p) break;
            cum += p;
            k2++;
            if (cum >= s->p.top_p) break;
        }
        if (k2 > 0) keep = k2;
    }

    /* 6) 重归一化 + 采样 */
    float total = 0.0f;
    for (int32_t i = 0; i < keep; i++) total += s->probs[s->idx[i]];
    if (total <= 0.0f) return s->idx[0];

    const float r = rand_f32(s) * total;
    float acc = 0.0f;
    for (int32_t i = 0; i < keep; i++) {
        acc += s->probs[s->idx[i]];
        if (acc >= r) return s->idx[i];
    }
    return s->idx[keep - 1];
}