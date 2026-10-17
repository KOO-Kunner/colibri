/* SPDX-License-Identifier: MIT */
#include "colibri/clb_ops.h"
#include <math.h>

/* 数值稳定的 in-place softmax */
void clb_softmax_inplace(float *x, int64_t n)
{
    if (n <= 0) return;

    float maxv = x[0];
    for (int64_t i = 1; i < n; i++)
        if (x[i] > maxv) maxv = x[i];

    float sum = 0.0f;
    for (int64_t i = 0; i < n; i++) {
        x[i] = expf(x[i] - maxv);
        sum += x[i];
    }

    const float inv = (sum > 0.0f) ? 1.0f / sum : 0.0f;
    for (int64_t i = 0; i < n; i++) x[i] *= inv;
}

/* SiLU (Swish): x / (1 + e^{-x}) */
void clb_silu_inplace(float *x, int64_t n)
{
    for (int64_t i = 0; i < n; i++)
        x[i] = x[i] / (1.0f + expf(-x[i]));
}

/* GELU (tanh 近似) */
void clb_gelu_inplace(float *x, int64_t n)
{
    const float k = 0.7978845608f; /* sqrt(2/pi) */
    for (int64_t i = 0; i < n; i++) {
        const float v = x[i];
        x[i] = 0.5f * v * (1.0f + tanhf(k * (v + 0.044715f * v * v * v)));
    }
}

/* 逐元素乘：a *= b */
void clb_mul_inplace(float *a, const float *b, int64_t n)
{
    for (int64_t i = 0; i < n; i++) a[i] *= b[i];
}

/* 逐元素加：a += b */
void clb_add_inplace(float *a, const float *b, int64_t n)
{
    for (int64_t i = 0; i < n; i++) a[i] += b[i];
}