/* SPDX-License-Identifier: MIT */
#include "colibri/clb_tensor.h"
#include "colibri/clb_alloc.h"
#include "colibri/clb_log.h"
#include "colibri/clb_quant.h"
#include <string.h>

static void compute_strides(clb_tensor *t)
{
    t->nb[0] = clb_dtype_block_bytes(t->dtype);
    for (int i = 1; i < CLB_MAX_DIMS; i++) {
        int64_t dim = (i - 1 < t->n_dims) ? t->ne[i - 1] : 1;
        if (clb_dtype_is_quantized(t->dtype) && i == 1) {
            t->nb[i] = clb_dtype_row_bytes(t->dtype, t->ne[0]);
        } else {
            t->nb[i] = t->nb[i - 1] * (size_t)dim;
        }
    }
}

clb_status clb_tensor_init(clb_tensor *t, const char *name,
                           clb_dtype dtype, int32_t n_dims, const int64_t *ne)
{
    if (!t || !name || n_dims < 1 || n_dims > CLB_MAX_DIMS) return CLB_ERR_INVALID_ARG;
    memset(t, 0, sizeof(*t));
    strncpy(t->name, name, CLB_TENSOR_NAME_MAX - 1);
    t->dtype  = dtype;
    t->n_dims = n_dims;
    for (int i = 0; i < CLB_MAX_DIMS; i++) t->ne[i] = (i < n_dims) ? ne[i] : 1;
    compute_strides(t);

    size_t row = clb_dtype_row_bytes(dtype, t->ne[0]);
    size_t total;
    if (!clb_size_mul(row, (size_t)(clb_tensor_nrows(t)), &total))
        return CLB_ERR_INVALID_ARG;
    t->nbytes = total;
    t->data   = NULL;
    t->owned  = false;
    return CLB_OK;
}

clb_status clb_tensor_alloc(clb_tensor *t, const char *name,
                            clb_dtype dtype, int32_t n_dims, const int64_t *ne)
{
    clb_status st = clb_tensor_init(t, name, dtype, n_dims, ne);
    if (st != CLB_OK) return st;

    t->data = clb_aligned_alloc(CLB_DEFAULT_ALIGN, t->nbytes ? t->nbytes : 1);
    if (!t->data) return CLB_ERR_OUT_OF_MEMORY;
    memset(t->data, 0, t->nbytes);
    t->owned = true;
    return CLB_OK;
}

void clb_tensor_free(clb_tensor *t)
{
    if (!t) return;
    if (t->owned && t->data) clb_aligned_free(t->data);
    t->data  = NULL;
    t->owned = false;
    t->nbytes = 0;
}

void clb_tensor_view(clb_tensor *dst, const clb_tensor *src, const char *name,
                     int32_t n_dims, const int64_t *ne)
{
    clb_tensor_init(dst, name, src->dtype, n_dims, ne);
    dst->data  = src->data;
    dst->owned = false;
}

int64_t clb_tensor_nelements(const clb_tensor *t)
{
    int64_t n = 1;
    for (int i = 0; i < t->n_dims; i++) n *= t->ne[i];
    return n;
}

int64_t clb_tensor_nrows(const clb_tensor *t)
{
    return t->n_dims >= 2 ? t->ne[1] : 1;
}

size_t clb_tensor_row_bytes(const clb_tensor *t)
{
    return t->nb[1] ? t->nb[1] : clb_dtype_row_bytes(t->dtype, t->ne[0]);
}

void *clb_tensor_row_raw(clb_tensor *t, int64_t row)
{
    return (uint8_t *)t->data + (size_t)row * clb_tensor_row_bytes(t);
}

const void *clb_tensor_row_raw_c(const clb_tensor *t, int64_t row)
{
    return (const uint8_t *)t->data + (size_t)row * clb_tensor_row_bytes(t);
}

float *clb_tensor_row_f32(clb_tensor *t, int64_t row)
{
    return (float *)clb_tensor_row_raw(t, row);
}

const float *clb_tensor_row_f32_c(const clb_tensor *t, int64_t row)
{
    return (const float *)clb_tensor_row_raw_c(t, row);
}

clb_status clb_tensor_to_f32(const clb_tensor *src, float *dst)
{
    if (!src || !dst) return CLB_ERR_INVALID_ARG;
    const int64_t k = src->ne[0];
    const int64_t rows = clb_tensor_nrows(src);

    for (int64_t r = 0; r < rows; r++) {
        const void *row = clb_tensor_row_raw_c(src, r);
        float *out = dst + r * k;
        switch (src->dtype) {
        case CLB_DTYPE_F32:  memcpy(out, row, (size_t)k * 4); break;
        case CLB_DTYPE_Q8_0: clb_dequantize_row_q8_0(row, out, k); break;
        case CLB_DTYPE_Q4_0: clb_dequantize_row_q4_0(row, out, k); break;
        default: return CLB_ERR_UNSUPPORTED;
        }
    }
    return CLB_OK;
}