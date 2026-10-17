/* SPDX-License-Identifier: MIT */
#include "clb_test.h"
#include "colibri/clb_tensor.h"
#include "colibri/clb_quant.h"

static void test_alloc_and_shape(void)
{
    int64_t ne[2] = { 128, 4 };
    clb_tensor t;
    clb_status st = clb_tensor_alloc(&t, "test_w", CLB_DTYPE_Q8_0, 2, ne);

    TEST_ASSERT(st == CLB_OK, "q8_0 tensor alloc");
    TEST_ASSERT(t.ne[0] == 128 && t.ne[1] == 4, "shape preserved");
    TEST_ASSERT(t.n_dims == 2, "n_dims=2");

    /* 每行 128 元素 / 32 = 4 块 * 34 字节 = 136 字节 */
    size_t expected_row = 4u * CLB_Q8_0_BYTES;
    TEST_ASSERT(t.nb[1] == expected_row, "q8_0 row stride");
    TEST_ASSERT(t.nbytes == expected_row * 4, "total bytes");
    TEST_ASSERT(t.data != NULL, "data allocated");
    TEST_ASSERT(t.owned == true, "owned flag set");

    clb_tensor_free(&t);
    TEST_ASSERT(t.data == NULL, "data freed");
}

static void test_f32_access(void)
{
    int64_t ne[1] = { 64 };
    clb_tensor v;
    clb_status st = clb_tensor_alloc(&v, "v", CLB_DTYPE_F32, 1, ne);
    TEST_ASSERT(st == CLB_OK, "f32 tensor alloc");

    float *row = clb_tensor_row_f32(&v, 0);
    for (int i = 0; i < 64; i++) row[i] = (float)i;

    const float *crow = clb_tensor_row_f32_c(&v, 0);
    TEST_ASSERT_NEAR(crow[0],  0.0f,  1e-6f, "row[0]");
    TEST_ASSERT_NEAR(crow[63], 63.0f, 1e-6f, "row[63]");
    TEST_ASSERT(clb_tensor_nelements(&v) == 64, "nelements");
    TEST_ASSERT(clb_tensor_nrows(&v) == 1, "nrows 1D");

    clb_tensor_free(&v);
}

static void test_quant_to_f32(void)
{
    /* 构造 64 元素 f32 张量，量化再反量化，误差应可控 */
    int64_t ne[1] = { 64 };
    clb_tensor tf;
    clb_tensor_alloc(&tf, "src", CLB_DTYPE_F32, 1, ne);
    float *p = clb_tensor_row_f32(&tf, 0);
    for (int i = 0; i < 64; i++) p[i] = sinf((float)i * 0.3f);

    /* 用 q8_0 手工量化 */
    uint8_t q[2 * CLB_Q8_0_BYTES];
    clb_quantize_row_q8_0(p, q, 64);
    float back[64];
    clb_dequantize_row_q8_0(q, back, 64);

    float max_err = 0.0f;
    for (int i = 0; i < 64; i++) {
        float e = fabsf(p[i] - back[i]);
        if (e > max_err) max_err = e;
    }
    TEST_ASSERT(max_err < 0.05f, "quant->dequant error bounded");

    clb_tensor_free(&tf);
}

void test_tensor_all(void)
{
    printf("== tensor ==\n");
    test_alloc_and_shape();
    test_f32_access();
    test_quant_to_f32();
}