/* SPDX-License-Identifier: MIT */
#ifndef CLB_QUANT_H
#define CLB_QUANT_H

#include "clb_types.h"

CLB_EXTERN_C_BEGIN

/* 量化（k 必须为块大小的整数倍） */
CLB_API void clb_quantize_row_q8_0(const float *src, void *dst, int64_t k);
CLB_API void clb_quantize_row_q4_0(const float *src, void *dst, int64_t k);

/* 反量化 */
CLB_API void clb_dequantize_row_q8_0(const void *src, float *dst, int64_t k);
CLB_API void clb_dequantize_row_q4_0(const void *src, float *dst, int64_t k);

/* 点积内核：n 为元素数 */
CLB_API float clb_vec_dot_f32   (int64_t n, const void *a, const void *b);
CLB_API float clb_vec_dot_q8_0  (int64_t n, const void *a, const void *b); /* a,b 均 q8_0 */
CLB_API float clb_vec_dot_q4_0_q8_0(int64_t n, const void *a, const void *b); /* a=q4_0, b=q8_0 */

/* 量化张量行字节数所需的块数 */
CLB_API int64_t clb_quant_nblocks(int64_t k);

CLB_EXTERN_C_END
#endif /* CLB_QUANT_H */