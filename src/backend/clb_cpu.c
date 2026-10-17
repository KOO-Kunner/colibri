/* SPDX-License-Identifier: MIT */
#include "colibri/clb_cpu.h"
#include <stdio.h>
#include <string.h>

#if defined(_MSC_VER)
#  include <intrin.h>
#elif defined(__x86_64__) || defined(__i386__)
#  include <cpuid.h>
#endif

#if !defined(_WIN32)
#  include <unistd.h>
#endif

static clb_cpu_features g_feat;
static int              g_detected = 0;

static void detect_once(void)
{
    memset(&g_feat, 0, sizeof(g_feat));

#if defined(__x86_64__) || defined(__i386__)
#  if defined(_MSC_VER)
    int regs[4];
    __cpuid(regs, 1);
    g_feat.sse2   = (regs[3] & (1 << 26)) != 0;
    g_feat.sse41  = (regs[2] & (1 << 19)) != 0;
    g_feat.avx    = (regs[2] & (1 << 28)) != 0;
    g_feat.fma    = (regs[2] & (1 << 12)) != 0;
    __cpuidex(regs, 7, 0);
    g_feat.avx2   = (regs[1] & (1 << 5)) != 0;
    g_feat.avx512f= (regs[1] & (1 << 16)) != 0;
#  else
    unsigned eax, ebx, ecx, edx;
    if (__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
        g_feat.sse2  = (edx & (1u << 26)) != 0;
        g_feat.sse41 = (ecx & (1u << 19)) != 0;
        g_feat.avx   = (ecx & (1u << 28)) != 0;
        g_feat.fma   = (ecx & (1u << 12)) != 0;
    }
    if (__get_cpuid_max(0, NULL) >= 7) {
        __cpuid_count(7, 0, eax, ebx, ecx, edx);
        g_feat.avx2    = (ebx & (1u << 5))  != 0;
        g_feat.avx512f = (ebx & (1u << 16)) != 0;
    }
#  endif
#elif defined(__aarch64__) || defined(__ARM_NEON)
    g_feat.neon = true;
#endif

    g_feat.cache_line_size = 64;

#if defined(_SC_NPROCESSORS_ONLN)
    long ncpu = sysconf(_SC_NPROCESSORS_ONLN);
    if (ncpu > 0) { g_feat.n_logical_cores = (int)ncpu; g_feat.n_physical_cores = (int)ncpu; }
#endif
    if (g_feat.n_logical_cores <= 0) { g_feat.n_logical_cores = 1; g_feat.n_physical_cores = 1; }

    g_detected = 1;
}

const clb_cpu_features *clb_cpu_detect(void)
{
    if (!g_detected) detect_once();
    return &g_feat;
}

int clb_cpu_default_threads(void)
{
    const clb_cpu_features *f = clb_cpu_detect();
    int n = f->n_physical_cores > 0 ? f->n_physical_cores : f->n_logical_cores;
    if (n < 1) n = 1;
    if (n > 32) n = 32;
    return n;
}

void clb_cpu_summary(char *buf, size_t buf_size)
{
    const clb_cpu_features *f = clb_cpu_detect();
    snprintf(buf, buf_size,
             "cores=%d/%d simd={sse2:%d sse41:%d avx:%d avx2:%d avx512f:%d fma:%d neon:%d}",
             f->n_physical_cores, f->n_logical_cores,
             f->sse2, f->sse41, f->avx, f->avx2, f->avx512f, f->fma, f->neon);
}