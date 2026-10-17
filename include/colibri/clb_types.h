/* SPDX-License-Identifier: MIT */
#ifndef CLB_TYPES_H
#define CLB_TYPES_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include "clb_export.h"

CLB_EXTERN_C_BEGIN

/* ---------- 数据类型 ---------- */
typedef enum clb_dtype {
    CLB_DTYPE_UNKNOWN = 0,
    CLB_DTYPE_F32     = 1,
    CLB_DTYPE_F16     = 2,
    CLB_DTYPE_Q8_0    = 3,   /* 32 元素 / 块：fp16 scale + 32×int8 */
    CLB_DTYPE_Q4_0    = 4    /* 32 元素 / 块：fp16 scale + 16×uint8(2×4bit) */
} clb_dtype;

#define CLB_QK8_0       32
#define CLB_Q8_0_BYTES  34   /* 2 + 32 */
#define CLB_QK4_0       32
#define CLB_Q4_0_BYTES  18   /* 2 + 16 */

/* ---------- fp16 ---------- */
typedef struct clb_fp16 { uint16_t bits; } clb_fp16;

CLB_API float    clb_fp16_to_fp32(clb_fp16 h);
CLB_API clb_fp16 clb_fp32_to_fp16(float f);

/* ---------- dtype 元信息 ---------- */
CLB_API size_t      clb_dtype_block_elems(clb_dtype t);  /* 每块元素数，F32 返回 1 */
CLB_API size_t      clb_dtype_block_bytes(clb_dtype t);  /* 每块字节数 */
CLB_API size_t      clb_dtype_row_bytes(clb_dtype t, int64_t ne0); /* 一行字节数 */
CLB_API const char *clb_dtype_name(clb_dtype t);
CLB_API bool        clb_dtype_is_quantized(clb_dtype t);

/* ---------- 量化块结构（内部布局，供内核使用） ---------- */
typedef struct clb_block_q8_0 { clb_fp16 d; int8_t  qs[CLB_QK8_0]; } clb_block_q8_0;
typedef struct clb_block_q4_0 { clb_fp16 d; uint8_t qs[CLB_QK4_0/2]; } clb_block_q4_0;

CLB_EXTERN_C_END
#endif /* CLB_TYPES_H */