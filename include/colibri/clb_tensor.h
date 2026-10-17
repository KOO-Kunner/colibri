/* SPDX-License-Identifier: MIT */
#ifndef CLB_TENSOR_H
#define CLB_TENSOR_H

#include "clb_types.h"
#include "clb_error.h"

CLB_EXTERN_C_BEGIN

#define CLB_MAX_DIMS     4
#define CLB_TENSOR_NAME_MAX 96

typedef struct clb_tensor {
    char      name[CLB_TENSOR_NAME_MAX];
    clb_dtype dtype;
    int32_t   n_dims;
    int64_t   ne[CLB_MAX_DIMS];
    size_t    nb[CLB_MAX_DIMS];
    void     *data;
    size_t    nbytes;
    bool      owned;
} clb_tensor;

/* ---- 生命周期 ---- */
CLB_API clb_status clb_tensor_init(clb_tensor *t, const char *name,
                                   clb_dtype dtype, int32_t n_dims,
                                   const int64_t *ne);
CLB_API clb_status clb_tensor_alloc(clb_tensor *t, const char *name,
                                    clb_dtype dtype, int32_t n_dims,
                                    const int64_t *ne);
CLB_API void       clb_tensor_free(clb_tensor *t);
CLB_API void       clb_tensor_view(clb_tensor *dst, const clb_tensor *src,
                                   const char *name, int32_t n_dims,
                                   const int64_t *ne);

/* ---- 属性 ---- */
CLB_API int64_t clb_tensor_nelements(const clb_tensor *t);
CLB_API int64_t clb_tensor_nrows(const clb_tensor *t);
CLB_API size_t  clb_tensor_row_bytes(const clb_tensor *t);

/* ---- 内存访问 ---- */
CLB_API float *clb_tensor_row_f32(clb_tensor *t, int64_t row);
CLB_API const float *clb_tensor_row_f32_c(const clb_tensor *t, int64_t row);
CLB_API void  *clb_tensor_row_raw(clb_tensor *t, int64_t row);
CLB_API const void *clb_tensor_row_raw_c(const clb_tensor *t, int64_t row);

/* ---- 转换 ---- */
CLB_API clb_status clb_tensor_to_f32(const clb_tensor *src, float *dst);

CLB_EXTERN_C_END
#endif /* CLB_TENSOR_H */