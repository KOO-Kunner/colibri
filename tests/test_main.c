/* SPDX-License-Identifier: MIT */
#include "clb_test.h"
#include "colibri/colibri.h"

/* 定义全局计数器（其他文件通过 extern 引用） */
int g_tests_run    = 0;
int g_tests_failed = 0;

void test_quant_all(void);
void test_tensor_all(void);
void test_sampler_all(void);

int main(void)
{
    clb_init();
    printf("colibri %s test suite\n", clb_version_string());
    printf("=====================================\n");

    test_quant_all();
    test_tensor_all();
    test_sampler_all();

    TEST_SUMMARY();
}