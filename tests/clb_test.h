/* SPDX-License-Identifier: MIT */
#ifndef CLB_TEST_H
#define CLB_TEST_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

extern int g_tests_run;
extern int g_tests_failed;

#define TEST_ASSERT(cond, msg)                                                 \
    do {                                                                       \
        g_tests_run++;                                                         \
        if (!(cond)) {                                                         \
            g_tests_failed++;                                                  \
            fprintf(stderr, "  [FAIL] %s:%d  %s\n", __FILE__, __LINE__, msg);  \
        }                                                                      \
    } while (0)

#define TEST_ASSERT_NEAR(a, b, eps, msg)                                       \
    TEST_ASSERT(fabs((double)((a) - (b))) <= (double)(eps), msg)

#define TEST_SUMMARY()                                                         \
    do {                                                                       \
        printf("\n%d tests, %d failed\n", g_tests_run, g_tests_failed);        \
        return g_tests_failed == 0 ? 0 : 1;                                    \
    } while (0)

#endif