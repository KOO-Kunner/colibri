/* SPDX-License-Identifier: MIT */
#include "colibri/clb_quant.h"
#include <math.h>
#include <string.h>

int64_t clb_quant_nblocks(int64_t k) { return (k + 31) / 32; }

void clb_quantize_row_q8_0(const float *src, void *dst, int64_t k)
{
    clb_block_q8_0 *y = (clb_block_q8_0 *)dst;
    const int64_t nb = k / CLB_QK8_0;
    for (int64_t i = 0; i < nb; i++) {
        float amax = 0.0f;
        for (int j = 0; j < CLB_QK8_0; j++) {
            const float a = fabsf(src[i * CLB_QK8_0 + j]);
            if (a > amax) amax = a;
        }
        const float d  = amax / 127.0f;
        const float id = d != 0.0f ? 1.0f / d : 0.0f;
        y[i].d = clb_fp32_to_fp16(d);
        for (int j = 0; j < CLB_QK8_0; j++) {
            float v = src[i * CLB_QK8_0 + j] * id;
            if (v >  127.0f) v =  127.0f;
            if (v < -127.0f) v = -127.0f;
            y[i].qs[j] = (int8_t)lrintf(v);
        }
    }
}

void clb_dequantize_row_q8_0(const void *src, float *dst, int64_t k)
{
    const clb_block_q8_0 *x = (const clb_block_q8_0 *)src;
    const int64_t nb = k / CLB_QK8_0;
    for (int64_t i = 0; i < nb; i++) {
        const float d = clb_fp16_to_fp32(x[i].d);
        for (int j = 0; j < CLB_QK8_0; j++)
            dst[i * CLB_QK8_0 + j] = (float)x[i].qs[j] * d;
    }
}

void clb_quantize_row_q4_0(const float *src, void *dst, int64_t k)
{
    clb_block_q4_0 *y = (clb_block_q4_0 *)dst;
    const int64_t nb = k / CLB_QK4_0;
    for (int64_t i = 0; i < nb; i++) {
        float amax = 0.0f, vmax = 0.0f;
        for (int j = 0; j < CLB_QK4_0; j++) {
            const float a = fabsf(src[i * CLB_QK4_0 + j]);
            if (a > amax) { amax = a; vmax = src[i * CLB_QK4_0 + j]; }
        }
        const float d  = vmax / -8.0f;
        const float id = d != 0.0f ? 1.0f / d : 0.0f;
        y[i].d = clb_fp32_to_fp16(d);
        for (int j = 0; j < CLB_QK4_0 / 2; j++) {
            const float x0 = src[i * CLB_QK4_0 + j]               * id;
            const float x1 = src[i * CLB_QK4_0 + j + CLB_QK4_0/2] * id;
            int q0 = (int)(x0 + 8.5f);
            int q1 = (int)(x1 + 8.5f);
            if (q0 < 0) q0 = 0;
            if (q0 > 15) q0 = 15;
            if (q1 < 0) q1 = 0;
            if (q1 > 15) q1 = 15;
            y[i].qs[j] = (uint8_t)(q0 | (q1 << 4));
        }
    }
}

void clb_dequantize_row_q4_0(const void *src, float *dst, int64_t k)
{
    const clb_block_q4_0 *x = (const clb_block_q4_0 *)src;
    const int64_t nb = k / CLB_QK4_0;
    for (int64_t i = 0; i < nb; i++) {
        const float d = clb_fp16_to_fp32(x[i].d);
        for (int j = 0; j < CLB_QK4_0 / 2; j++) {
            const int x0 = (int)(x[i].qs[j] & 0x0F) - 8;
            const int x1 = (int)(x[i].qs[j] >> 4)   - 8;
            dst[i * CLB_QK4_0 + j]                 = (float)x0 * d;
            dst[i * CLB_QK4_0 + j + CLB_QK4_0 / 2] = (float)x1 * d;
        }
    }
}

float clb_vec_dot_f32(int64_t n, const void *a, const void *b)
{
    const float *x = (const float *)a;
    const float *y = (const float *)b;
    float sum = 0.0f;
    int64_t i = 0;
    for (; i + 4 <= n; i += 4)
        sum += x[i]*y[i] + x[i+1]*y[i+1] + x[i+2]*y[i+2] + x[i+3]*y[i+3];
    for (; i < n; i++) sum += x[i] * y[i];
    return sum;
}

float clb_vec_dot_q8_0(int64_t n, const void *a, const void *b)
{
    const clb_block_q8_0 *x = (const clb_block_q8_0 *)a;
    const clb_block_q8_0 *y = (const clb_block_q8_0 *)b;
    const int64_t nb = n / CLB_QK8_0;
    float sumf = 0.0f;
    for (int64_t i = 0; i < nb; i++) {
        int sumi = 0;
        for (int j = 0; j < CLB_QK8_0; j++)
            sumi += (int)x[i].qs[j] * (int)y[i].qs[j];
        sumf += clb_fp16_to_fp32(x[i].d) * clb_fp16_to_fp32(y[i].d) * (float)sumi;
    }
    return sumf;
}

float clb_vec_dot_q4_0_q8_0(int64_t n, const void *a, const void *b)
{
    const clb_block_q4_0 *x = (const clb_block_q4_0 *)a;
    const clb_block_q8_0 *y = (const clb_block_q8_0 *)b;
    const int64_t nb = n / CLB_QK4_0;
    float sumf = 0.0f;
    for (int64_t i = 0; i < nb; i++) {
        const float dx = clb_fp16_to_fp32(x[i].d);
        const float dy = clb_fp16_to_fp32(y[i].d);
        int sumi = 0;
        for (int j = 0; j < CLB_QK4_0 / 2; j++) {
            const int v0 = (int)(x[i].qs[j] & 0x0F) - 8;
            const int v1 = (int)(x[i].qs[j] >> 4)   - 8;
            sumi += v0 * (int)y[i].qs[j] + v1 * (int)y[i].qs[j + CLB_QK4_0 / 2];
        }
        sumf += dx * dy * (float)sumi;
    }
    return sumf;
}