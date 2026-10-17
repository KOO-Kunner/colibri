/* SPDX-License-Identifier: MIT */
#include "colibri/clb_ops.h"
#include <math.h>

/*
 * 对形如 [n_heads, head_dim] 的向量做 RoPE。
 * 采用 GPT-NeoX 风格：相邻两维构成一个旋转对。
 *
 *   vec        : [n_heads * head_dim] f32
 *   n_dims     : n_heads * head_dim
 *   n_heads    : 头数（或 KV 头数）
 *   pos        : 当前位置
 *   theta      : RoPE 基频（通常 10000.0）
 *   freq_scale : 频率缩放因子（默认 1.0）
 */
void clb_rope_apply(float *vec, int64_t n_dims, int64_t n_heads,
                    int64_t pos, float theta, float freq_scale)
{
    if (n_heads <= 0) return;
    const int64_t head_dim = n_dims / n_heads;
    const int64_t half     = head_dim / 2;

    for (int64_t h = 0; h < n_heads; h++) {
        float *v = vec + h * head_dim;
        for (int64_t i = 0; i < half; i++) {
            const float freq = 1.0f /
                powf(theta, (float)(2 * i) / (float)head_dim);
            const float angle = (float)pos * freq * freq_scale;
            const float c = cosf(angle);
            const float s = sinf(angle);

            const float x0 = v[2 * i];
            const float x1 = v[2 * i + 1];

            v[2 * i]     = x0 * c - x1 * s;
            v[2 * i + 1] = x0 * s + x1 * c;
        }
    }
}