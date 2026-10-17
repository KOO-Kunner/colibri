/* SPDX-License-Identifier: MIT */
#include "clb_test.h"
#include "colibri/clb_sampler.h"

static void test_greedy(void)
{
    clb_sampler_params p = clb_sampler_default_params();
    p.n_vocab = 8;
    p.mode    = CLB_SAMPLER_GREEDY;

    clb_status st;
    clb_sampler *s = clb_sampler_create(&p, &st);
    TEST_ASSERT(s != NULL, "greedy sampler create");

    float logits[8] = { 0.1f, 0.2f, 5.0f, 0.0f, -1.0f, 0.3f, 0.4f, 0.2f };
    int32_t best = clb_sampler_sample(s, logits);
    TEST_ASSERT(best == 2, "greedy argmax=2");

    /* 第二次调用应该返回同样的结果（无历史惩罚时） */
    float logits2[8] = { 0.1f, 0.2f, 5.0f, 0.0f, -1.0f, 0.3f, 0.4f, 0.2f };
    int32_t best2 = clb_sampler_sample(s, logits2);
    TEST_ASSERT(best2 == 2, "greedy deterministic");

    clb_sampler_free(s);
}

static void test_top_k_1(void)
{
    /* top_k=1 应当退化为 argmax */
    clb_sampler_params p = clb_sampler_default_params();
    p.n_vocab        = 8;
    p.mode           = CLB_SAMPLER_TEMP;
    p.temp           = 1.0f;
    p.top_k          = 1;
    p.top_p          = 1.0f;
    p.min_p          = 0.0f;
    p.repeat_penalty = 1.0f;

    clb_status st;
    clb_sampler *s = clb_sampler_create(&p, &st);
    TEST_ASSERT(s != NULL, "top_k=1 sampler create");

    float logits[8] = { 0.1f, 0.2f, 5.0f, 0.0f, -1.0f, 0.3f, 0.4f, 0.2f };
    int32_t r = clb_sampler_sample(s, logits);
    TEST_ASSERT(r == 2, "top_k=1 -> argmax");

    clb_sampler_free(s);
}

static void test_repeat_penalty(void)
{
    /* 高重复惩罚应把历史 token 的分数压低 */
    clb_sampler_params p = clb_sampler_default_params();
    p.n_vocab        = 4;
    p.mode           = CLB_SAMPLER_GREEDY;
    p.repeat_penalty = 10.0f;
    p.repeat_last_n  = 4;

    clb_status st;
    clb_sampler *s = clb_sampler_create(&p, &st);
    TEST_ASSERT(s != NULL, "penalty sampler create");

    /* 先"生成" token 0 */
    clb_sampler_accept(s, 0);

    /* token 0 分数最高，但有惩罚后应该被选其它 token */
    float logits[4] = { 5.0f, 4.0f, 3.0f, 2.0f };
    int32_t r = clb_sampler_sample(s, logits);
    TEST_ASSERT(r != 0, "repeat penalty avoids token 0");

    clb_sampler_free(s);
}

static void test_seed_determinism(void)
{
    /* 相同 seed 的两次采样应产生相同结果 */
    clb_sampler_params p = clb_sampler_default_params();
    p.n_vocab        = 16;
    p.mode           = CLB_SAMPLER_TEMP;
    p.temp           = 1.0f;
    p.top_k          = 0;   /* 关闭 top-k，保留全部 */
    p.top_p          = 1.0f;
    p.min_p          = 0.0f;
    p.repeat_penalty = 1.0f;
    p.seed           = 12345;

    clb_status st;
    clb_sampler *s1 = clb_sampler_create(&p, &st);
    clb_sampler *s2 = clb_sampler_create(&p, &st);
    TEST_ASSERT(s1 && s2, "seeded samplers created");

    float logits[16];
    for (int i = 0; i < 16; i++) logits[i] = (float)i * 0.1f;

    float copy1[16], copy2[16];
    memcpy(copy1, logits, sizeof(logits));
    memcpy(copy2, logits, sizeof(logits));

    int32_t r1 = clb_sampler_sample(s1, copy1);
    int32_t r2 = clb_sampler_sample(s2, copy2);
    TEST_ASSERT(r1 == r2, "same seed same first sample");

    clb_sampler_free(s1);
    clb_sampler_free(s2);
}

void test_sampler_all(void)
{
    printf("== sampler ==\n");
    test_greedy();
    test_top_k_1();
    test_repeat_penalty();
    test_seed_determinism();
}