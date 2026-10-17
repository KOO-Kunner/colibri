/* SPDX-License-Identifier: MIT */
#include "colibri/clb_ops.h"
#include <math.h>

/* RMSNorm: y = x / sqrt(mean(x^2)+eps) * w */
void clb_rmsnorm(float *y, const float *x, const float *w,
                 int64_t n, float eps)
{
    float ss = 0.0f;
    for (int64_t i = 0; i < n; i++) ss += x[i] * x[i];
    ss /= (float)n;

    const float scale = 1.0f / sqrtf(ss + eps);
    for (int64_t i = 0; i < n; i++)
        y[i] = x[i] * scale * (w ? w[i] : 1.0f);
}

/* LayerNorm: y = (x - mean) / sqrt(var + eps) * w + b */
void clb_layernorm(float *y, const float *x, const float *w, const float *b,
                   int64_t n, float eps)
{
    float mean = 0.0f;
    for (int64_t i = 0; i < n; i++) mean += x[i];
    mean /= (float)n;

    float var = 0.0f;
    for (int64_t i = 0; i < n; i++) {
        const float d = x[i] - mean;
        var += d * d;
    }
    var /= (float)n;

    const float inv = 1.0f / sqrtf(var + eps);
    for (int64_t i = 0; i < n; i++) {
        float v = (x[i] - mean) * inv;
        if (w) v *= w[i];
        if (b) v += b[i];
        y[i] = v;
    }
}