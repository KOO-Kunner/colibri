/* SPDX-License-Identifier: MIT */
#define _POSIX_C_SOURCE 200809L

#include "colibri/clb_threadpool.h"
#include "colibri/clb_log.h"
#include "colibri/clb_alloc.h"
#include <pthread.h>
#include <string.h>

struct clb_threadpool {
    int              n_threads;   /* 总线程数（含主线程） */
    pthread_t       *threads;     /* 工作线程句柄，数量 = n_threads - 1 */
    pthread_mutex_t  mutex;
    pthread_cond_t   wake;
    pthread_cond_t   done;
    clb_task_fn      fn;
    void            *arg;
    int64_t          n_tasks;
    int64_t          next_task;
    int64_t          n_done;
    uint64_t         generation;
    bool             shutdown;
};

/* -------- 工作线程主循环 -------- */
static void *worker_main(void *p)
{
    clb_threadpool *tp = (clb_threadpool *)p;
    uint64_t gen = 0;

    pthread_mutex_lock(&tp->mutex);
    for (;;) {
        while (!tp->shutdown && tp->generation == gen)
            pthread_cond_wait(&tp->wake, &tp->mutex);

        if (tp->shutdown) break;

        gen = tp->generation;
        const int nthreads = tp->n_threads;

        for (;;) {
            if (tp->next_task >= tp->n_tasks) break;
            const int64_t ith = tp->next_task++;
            clb_task_fn fn = tp->fn;
            void *arg = tp->arg;

            pthread_mutex_unlock(&tp->mutex);
            fn((int)ith, nthreads, arg);
            pthread_mutex_lock(&tp->mutex);

            tp->n_done++;
            if (tp->n_done == tp->n_tasks)
                pthread_cond_signal(&tp->done);
        }
    }
    pthread_mutex_unlock(&tp->mutex);
    return NULL;
}

/* -------- 创建 -------- */
clb_threadpool *clb_threadpool_create(int n_threads, clb_status *st)
{
    if (n_threads < 1) n_threads = 1;

    clb_threadpool *tp = (clb_threadpool *)clb_calloc(1, sizeof(*tp));
    if (!tp) { if (st) *st = CLB_ERR_OUT_OF_MEMORY; return NULL; }

    tp->n_threads = n_threads;

    if (pthread_mutex_init(&tp->mutex, NULL) != 0) goto fail;
    if (pthread_cond_init(&tp->wake, NULL) != 0) goto fail_mutex;
    if (pthread_cond_init(&tp->done, NULL) != 0) goto fail_wake;

    if (n_threads > 1) {
        tp->threads = (pthread_t *)clb_calloc((size_t)(n_threads - 1), sizeof(pthread_t));
        if (!tp->threads) goto fail_done;

        for (int i = 0; i < n_threads - 1; i++) {
            if (pthread_create(&tp->threads[i], NULL, worker_main, tp) != 0) {
                CLB_LOGW("pthread_create failed at %d, degrade to %d threads",
                         i, i + 1);
                tp->n_threads = i + 1;
                break;
            }
        }
    }

    if (st) *st = CLB_OK;
    return tp;

fail_done:
    pthread_cond_destroy(&tp->done);
fail_wake:
    pthread_cond_destroy(&tp->wake);
fail_mutex:
    pthread_mutex_destroy(&tp->mutex);
fail:
    clb_free(tp);
    if (st) *st = CLB_ERR_THREAD;
    return NULL;
}

/* -------- 销毁 -------- */
void clb_threadpool_destroy(clb_threadpool *tp)
{
    if (!tp) return;

    pthread_mutex_lock(&tp->mutex);
    tp->shutdown = true;
    pthread_cond_broadcast(&tp->wake);
    pthread_mutex_unlock(&tp->mutex);

    for (int i = 0; i < tp->n_threads - 1; i++)
        pthread_join(tp->threads[i], NULL);

    clb_free(tp->threads);
    pthread_cond_destroy(&tp->done);
    pthread_cond_destroy(&tp->wake);
    pthread_mutex_destroy(&tp->mutex);
    clb_free(tp);
}

int clb_threadpool_n_threads(const clb_threadpool *tp)
{
    return tp ? tp->n_threads : 1;
}

/* -------- 并行任务分发 -------- */
void clb_threadpool_parallel(clb_threadpool *tp, int64_t n,
                             clb_task_fn fn, void *arg)
{
    if (n <= 0) return;

    /* 单线程或单任务：退化为串行，避免线程同步开销 */
    if (!tp || tp->n_threads <= 1 || n == 1) {
        for (int64_t i = 0; i < n; i++) fn((int)i, 1, arg);
        return;
    }

    pthread_mutex_lock(&tp->mutex);
    tp->fn        = fn;
    tp->arg       = arg;
    tp->n_tasks   = n;
    tp->next_task = 0;
    tp->n_done    = 0;
    tp->generation++;
    pthread_cond_broadcast(&tp->wake);

    /* 主线程也参与干活 */
    for (;;) {
        int64_t ith = -1;
        if (tp->next_task < tp->n_tasks) {
            ith = tp->next_task++;
        } else {
            break;
        }

        pthread_mutex_unlock(&tp->mutex);
        fn((int)ith, tp->n_threads, arg);
        pthread_mutex_lock(&tp->mutex);

        tp->n_done++;
        if (tp->n_done == tp->n_tasks)
            pthread_cond_broadcast(&tp->done);
    }

    while (tp->n_done < tp->n_tasks)
        pthread_cond_wait(&tp->done, &tp->mutex);

    pthread_mutex_unlock(&tp->mutex);
}