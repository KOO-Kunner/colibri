/* SPDX-License-Identifier: MIT */
#include "clb_test.h"
#include "colibri/clb_quant.h"

void test_fp16_roundtrip(void);
void test_q8_0_roundtrip(void);
void test_q4_0_roundtrip(void);
void test_dot_consistency(void);

void test_fp16_roundtrip(void)
{
    const float vals[] = { 0.0f, 1.0f, -1.0f, 0.5f, 3.14159f,
                           1e-4f, 65504.0f, -0.125f };
    for (size_t i = 0; i < sizeof(vals)/sizeof(vals[0]); i++) {
        float f = vals[i];
        clb_fp16 h = clb_fp32_to_fp16(f);
        float back = clb_fp16_to_fp32(h);
        TEST_ASSERT_NEAR(back, f, fabsf(f) * 0.01f + 1e-6f, "fp16 roundtrip");
    }
}

void test_q8_0_roundtrip(void)
{
    float src[64];
    for (int i = 0; i < 64; i++) src[i] = sinf((float)i * 0.3f) * 2.0f;

    uint8_t q[2 * CLB_Q8_0_BYTES];
    float dst[64];
    clb_quantize_row_q8_0(src, q, 64);
    clb_dequantize_row_q8_0(q, dst, 64);

    float max_err = 0.0f;
    for (int i = 0; i < 64; i++) {
        float e = fabsf(src[i] - dst[i]);
        if (e > max_err) max_err = e;
    }
    TEST_ASSERT(max_err < 0.05f, "q8_0 roundtrip error too large");
}

void test_q4_0_roundtrip(void)
{
    float src[32];
    for (int i = 0; i < 32; i++) src[i] = cosf((float)i * 0.5f);

    uint8_t q[CLB_Q4_0_BYTES];
    float dst[32];
    clb_quantize_row_q4_0(src, q, 32);
    clb_dequantize_row_q4_0(q, dst, 32);

    float max_err = 0.0f;
    for (int i = 0; i < 32; i++) {
        float e = fabsf(src[i] - dst[i]);
        if (e > max_err) max_err = e;
    }
    TEST_ASSERT(max_err < 0.2f, "q4_0 roundtrip error too large");
}

void test_dot_consistency(void)
{
    float a[32], b[32];
    for (int i = 0; i < 32; i++) {
        a[i] = (float)(i % 7) - 3.0f;
        b[i] = (float)(i % 5) - 2.0f;
    }

    float ref = clb_vec_dot_f32(32, a, b);

    uint8_t qa[CLB_Q8_0_BYTES], qb[CLB_Q8_0_BYTES];
    clb_quantize_row_q8_0(a, qa, 32);
    clb_quantize_row_q8_0(b, qb, 32);
    float approx = clb_vec_dot_q8_0(32, qa, qb);

    TEST_ASSERT_NEAR(approx, ref, fabsf(ref) * 0.1f + 0.5f, "q8_0 dot ~= f32");
}

void test_quant_all(void)
{
    printf("== quant ==\n");
    test_fp16_roundtrip();
    test_q8_0_roundtrip();
    test_q4_0_roundtrip();
    test_dot_consistency();
}