/* SPDX-License-Identifier: MIT */
#ifndef CLB_OPS_H
#define CLB_OPS_H

#include "clb_tensor.h"
#include "clb_threadpool.h"

CLB_EXTERN_C_BEGIN

CLB_API clb_status clb_matmul(clb_tensor *dst, const clb_tensor *w,
                              const clb_tensor *src, clb_threadpool *tp);

CLB_API void clb_rmsnorm(float *y, const float *x, const float *w,
                         int64_t n, float eps);
CLB_API void clb_layernorm(float *y, const float *x, const float *w,
                           const float *b, int64_t n, float eps);
CLB_API void clb_softmax_inplace(float *x, int64_t n);
CLB_API void clb_silu_inplace(float *x, int64_t n);
CLB_API void clb_gelu_inplace(float *x, int64_t n);
CLB_API void clb_mul_inplace(float *a, const float *b, int64_t n);
CLB_API void clb_add_inplace(float *a, const float *b, int64_t n);
CLB_API void clb_rope_apply(float *vec, int64_t n_dims, int64_t n_heads,
                            int64_t pos, float theta, float freq_scale);

CLB_EXTERN_C_END
#endif /* CLB_OPS_H */