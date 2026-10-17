/* SPDX-License-Identifier: MIT */
#ifndef CLB_THREADPOOL_H
#define CLB_THREADPOOL_H

#include <stdint.h>
#include "clb_error.h"

CLB_EXTERN_C_BEGIN

typedef struct clb_threadpool clb_threadpool;

/* 任务函数：ith ∈ [0, nthreads)，arg 为用户指针 */
typedef void (*clb_task_fn)(int ith, int nthreads, void *arg);

CLB_API clb_threadpool *clb_threadpool_create(int n_threads, clb_status *st);
CLB_API void            clb_threadpool_destroy(clb_threadpool *tp);
CLB_API int             clb_threadpool_n_threads(const clb_threadpool *tp);

/* 将 [0, n) 划分为若干任务并行执行；n==1 或单线程时退化为串行 */
CLB_API void clb_threadpool_parallel(clb_threadpool *tp, int64_t n,
                                     clb_task_fn fn, void *arg);

CLB_EXTERN_C_END
#endif /* CLB_THREADPOOL_H */