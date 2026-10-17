/* SPDX-License-Identifier: MIT */
#include "colibri/clb_ops.h"
#include "colibri/clb_quant.h"
#include "colibri/clb_alloc.h"
#include "colibri/clb_log.h"
#include <string.h>

/*
 * 矩阵乘：dst[m, n] = W[n, k] · src[m, k]^T
 *   W   : ne[0]=k, ne[1]=n
 *   src : ne[0]=k, ne[1]=m
 *   dst : ne[0]=n, ne[1]=m  (f32)
 */

typedef struct {
    clb_tensor       *dst;
    const clb_tensor *w;
    const clb_tensor *src;
    int64_t           m;
    int64_t           n;
    int64_t           k;
    uint8_t          *src_q8;   /* src 行的 q8_0 量化缓冲，[m][k] */
} mm_ctx;

static float dot_row(const clb_tensor *w, int64_t j,
                     const uint8_t *src_q8_row, const float *src_f32_row,
                     int64_t k)
{
    const void *wrow = clb_tensor_row_raw_c(w, j);
    switch (w->dtype) {
    case CLB_DTYPE_F32:
        return clb_vec_dot_f32(k, wrow, src_f32_row);
    case CLB_DTYPE_Q8_0:
        return clb_vec_dot_q8_0(k, wrow, src_q8_row);
    case CLB_DTYPE_Q4_0:
        return clb_vec_dot_q4_0_q8_0(k, wrow, src_q8_row);
    default:
        return 0.0f;
    }
}

static void mm_worker(int ith, int nthreads, void *arg)
{
    mm_ctx *c = (mm_ctx *)arg;
    const size_t src_q8_row_bytes =
        (size_t)(c->k / 32) * CLB_Q8_0_BYTES;

    for (int64_t i = ith; i < c->m; i += nthreads) {
        const float   *src_f32_row = clb_tensor_row_f32_c(c->src, i);
        const uint8_t *src_q8_row  = c->src_q8
                                   ? c->src_q8 + (size_t)i * src_q8_row_bytes
                                   : NULL;
        float         *dst_row     = clb_tensor_row_f32(c->dst, i);

        for (int64_t j = 0; j < c->n; j++)
            dst_row[j] = dot_row(c->w, j, src_q8_row, src_f32_row, c->k);
    }
}

clb_status clb_matmul(clb_tensor *dst, const clb_tensor *w,
                      const clb_tensor *src, clb_threadpool *tp)
{
    if (!dst || !w || !src) return CLB_ERR_INVALID_ARG;
    if (dst->dtype != CLB_DTYPE_F32) return CLB_ERR_INVALID_ARG;

    const int64_t k  = src->ne[0];
    const int64_t m  = clb_tensor_nrows(src);
    const int64_t n  = w->ne[1];
    const int64_t wk = w->ne[0];

    if (wk != k) {
        CLB_LOGE("matmul dim mismatch: w.ne0=%lld src.ne0=%lld",
                 (long long)wk, (long long)k);
        return CLB_ERR_INVALID_ARG;
    }
    if (dst->ne[0] < n || clb_tensor_nrows(dst) < m)
        return CLB_ERR_INVALID_ARG;

    mm_ctx c = { dst, w, src, m, n, k, NULL };

    /* 量化权重需要对应的 q8_0 激活 */
    if (clb_dtype_is_quantized(w->dtype)) {
        if (k % 32 != 0) return CLB_ERR_UNSUPPORTED;

        size_t row_bytes = (size_t)(k / 32) * CLB_Q8_0_BYTES;
        size_t total;
        if (!clb_size_mul(row_bytes, (size_t)m, &total))
            return CLB_ERR_OUT_OF_MEMORY;

        c.src_q8 = (uint8_t *)clb_aligned_alloc(CLB_DEFAULT_ALIGN, total);
        if (!c.src_q8) return CLB_ERR_OUT_OF_MEMORY;

        for (int64_t i = 0; i < m; i++)
            clb_quantize_row_q8_0(clb_tensor_row_f32_c(src, i),
                                  c.src_q8 + (size_t)i * row_bytes, k);
    }

    clb_threadpool_parallel(tp, m, mm_worker, &c);

    if (c.src_q8) clb_aligned_free(c.src_q8);
    return CLB_OK;
}