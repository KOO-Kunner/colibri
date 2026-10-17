/* SPDX-License-Identifier: MIT */
#include "colibri/clb_types.h"
#include <string.h>

float clb_fp16_to_fp32(clb_fp16 h)
{
    uint32_t sign = (uint32_t)(h.bits >> 15) << 31;
    uint32_t exp  = (h.bits >> 10) & 0x1Fu;
    uint32_t man  = h.bits & 0x3FFu;
    uint32_t f32;

    if (exp == 0) {
        if (man == 0) { f32 = sign; }
        else {
            uint32_t m = man;
            int shift = 0;
            while ((m & 0x400u) == 0) { m <<= 1; shift++; }
            m &= 0x3FFu;
            int e = 113 - shift;
            f32 = sign | ((uint32_t)e << 23) | (m << 13);
        }
    } else if (exp == 0x1Fu) {
        f32 = sign | 0x7F800000u | (man << 13);
    } else {
        f32 = sign | ((exp - 15u + 127u) << 23) | (man << 13);
    }

    float out;
    memcpy(&out, &f32, sizeof(out));
    return out;
}

clb_fp16 clb_fp32_to_fp16(float f)
{
    uint32_t x;
    memcpy(&x, &f, sizeof(x));

    uint16_t sign = (uint16_t)((x >> 16) & 0x8000u);
    int32_t  exp  = (int32_t)((x >> 23) & 0xFFu) - 127;
    uint32_t man  = x & 0x7FFFFFu;
    uint16_t out;

    if (exp == 128) { out = (uint16_t)(sign | 0x7C00u | (man ? 0x200u : 0)); }
    else if (exp > 15) { out = (uint16_t)(sign | 0x7C00u); }
    else if (exp >= -14) { out = (uint16_t)(sign | ((uint16_t)(exp + 15) << 10) | (uint16_t)(man >> 13)); }
    else if (exp >= -24) {
        uint32_t m = (0x800000u | man) >> (-(exp + 1));
        out = (uint16_t)(sign | (uint16_t)m);
    } else { out = sign; }

    clb_fp16 h; h.bits = out; return h;
}

size_t clb_dtype_block_elems(clb_dtype t)
{
    switch (t) {
    case CLB_DTYPE_Q8_0: return CLB_QK8_0;
    case CLB_DTYPE_Q4_0: return CLB_QK4_0;
    default:             return 1;
    }
}

size_t clb_dtype_block_bytes(clb_dtype t)
{
    switch (t) {
    case CLB_DTYPE_F32:  return 4;
    case CLB_DTYPE_F16:  return 2;
    case CLB_DTYPE_Q8_0: return CLB_Q8_0_BYTES;
    case CLB_DTYPE_Q4_0: return CLB_Q4_0_BYTES;
    default:             return 0;
    }
}

size_t clb_dtype_row_bytes(clb_dtype t, int64_t ne0)
{
    if (ne0 <= 0) return 0;
    size_t blk = clb_dtype_block_elems(t);
    size_t bpb = clb_dtype_block_bytes(t);
    if (blk == 0) return 0;
    size_t nblocks = (size_t)((ne0 + (int64_t)blk - 1) / (int64_t)blk);
    return nblocks * bpb;
}

const char *clb_dtype_name(clb_dtype t)
{
    switch (t) {
    case CLB_DTYPE_F32:  return "f32";
    case CLB_DTYPE_F16:  return "f16";
    case CLB_DTYPE_Q8_0: return "q8_0";
    case CLB_DTYPE_Q4_0: return "q4_0";
    default:             return "unknown";
    }
}

bool clb_dtype_is_quantized(clb_dtype t)
{
    return t == CLB_DTYPE_Q8_0 || t == CLB_DTYPE_Q4_0;
}