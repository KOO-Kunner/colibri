/* SPDX-License-Identifier: MIT */
#include "colibri/colibri.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

static void print_usage(const char *prog)
{
    printf(
        "colibri %s — 纯 C 轻量大模型推理引擎\n"
        "\n"
        "用法:\n"
        "  %s --model <path.clb> [options]\n"
        "\n"
        "核心选项:\n"
        "  -m, --model <path>       模型文件路径（必需）\n"
        "  -p, --prompt <text>      输入 prompt\n"
        "  -n, --n-predict <int>    最大生成 token 数（默认 128，-1=至 EOG）\n"
        "  -c, --ctx <int>          上下文长度（默认取模型训练长度）\n"
        "  -t, --threads <int>      线程数（默认自动探测）\n"
        "  -s, --seed <uint>        随机种子\n"
        "\n"
        "采样选项:\n"
        "      --temp <float>       温度（默认 0.8；0 等价于 greedy）\n"
        "      --top-k <int>        top-k（默认 40，0=关闭）\n"
        "      --top-p <float>      top-p（默认 0.95）\n"
        "      --min-p <float>      min-p（默认 0.0）\n"
        "      --repeat-penalty <f> 重复惩罚（默认 1.1）\n"
        "\n"
        "其他:\n"
        "      --no-mmap            禁用内存映射\n"
        "      --echo-prompt        回显 prompt\n"
        "      --selftest           运行内部自检（不需要模型文件）\n"
        "  -v, --verbose            提升日志级别\n"
        "  -h, --help               显示帮助\n"
        "      --version            显示版本\n",
        CLB_VERSION_STRING, prog);
}

static bool stream_cb(const char *piece, int32_t token, void *user)
{
    (void)token; (void)user;
    fputs(piece, stdout);
    fflush(stdout);
    return true;
}

/* 内部自检：验证量化内核与基础算子 */
static int run_selftest(void)
{
    printf("=== colibri selftest ===\n");
    printf("version: %s\n", clb_version_string());

    char cpu_buf[256];
    clb_cpu_summary(cpu_buf, sizeof(cpu_buf));
    printf("cpu:     %s\n", cpu_buf);

    /* 量化 roundtrip */
    float src[64];
    for (int i = 0; i < 64; i++) src[i] = (float)(i - 32) * 0.1f;

    uint8_t q8[2 * CLB_Q8_0_BYTES];
    float   back[64];
    clb_quantize_row_q8_0(src, q8, 64);
    clb_dequantize_row_q8_0(q8, back, 64);

    float max_err = 0.0f;
    for (int i = 0; i < 64; i++) {
        float e = src[i] - back[i];
        if (e < 0) e = -e;
        if (e > max_err) max_err = e;
    }
    printf("q8_0 roundtrip: max_err=%.6f\n", (double)max_err);

    /* softmax */
    float sm[4] = { 1.0f, 2.0f, 3.0f, 4.0f };
    clb_softmax_inplace(sm, 4);
    printf("softmax: %.4f %.4f %.4f %.4f\n",
           (double)sm[0], (double)sm[1], (double)sm[2], (double)sm[3]);

    /* rmsnorm */
    float x[4] = { 1.0f, 2.0f, 3.0f, 4.0f };
    float w[4] = { 1.0f, 1.0f, 1.0f, 1.0f };
    float y[4];
    clb_rmsnorm(y, x, w, 4, 1e-5f);
    printf("rmsnorm: %.4f %.4f %.4f %.4f\n",
           (double)y[0], (double)y[1], (double)y[2], (double)y[3]);

    printf("selftest OK\n");
    return 0;
}

int main(int argc, char **argv)
{
    clb_engine_params ep = clb_engine_default_params();
    clb_gen_params    gp = clb_gen_default_params();
    bool verbose   = false;
    bool selftest  = false;

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        #define NEXT() ((i + 1 < argc) ? argv[++i] : NULL)

        if      (!strcmp(a, "-m") || !strcmp(a, "--model"))   ep.model_path = NEXT();
        else if (!strcmp(a, "-p") || !strcmp(a, "--prompt"))  gp.prompt = NEXT();
        else if (!strcmp(a, "-n") || !strcmp(a, "--n-predict"))
            gp.n_predict = (int32_t)strtol(NEXT(), NULL, 10);
        else if (!strcmp(a, "-c") || !strcmp(a, "--ctx"))
            ep.n_ctx = (int32_t)strtol(NEXT(), NULL, 10);
        else if (!strcmp(a, "-t") || !strcmp(a, "--threads"))
            ep.n_threads = (int)strtol(NEXT(), NULL, 10);
        else if (!strcmp(a, "-s") || !strcmp(a, "--seed"))
            gp.sampler.seed = strtoull(NEXT(), NULL, 10);
        else if (!strcmp(a, "--temp"))
            gp.sampler.temp = strtof(NEXT(), NULL);
        else if (!strcmp(a, "--top-k"))
            gp.sampler.top_k = (int32_t)strtol(NEXT(), NULL, 10);
        else if (!strcmp(a, "--top-p"))
            gp.sampler.top_p = strtof(NEXT(), NULL);
        else if (!strcmp(a, "--min-p"))
            gp.sampler.min_p = strtof(NEXT(), NULL);
        else if (!strcmp(a, "--repeat-penalty"))
            gp.sampler.repeat_penalty = strtof(NEXT(), NULL);
        else if (!strcmp(a, "--no-mmap"))     ep.use_mmap = false;
        else if (!strcmp(a, "--echo-prompt")) gp.echo_prompt = true;
        else if (!strcmp(a, "--selftest"))    selftest = true;
        else if (!strcmp(a, "-v") || !strcmp(a, "--verbose")) verbose = true;
        else if (!strcmp(a, "-h") || !strcmp(a, "--help")) { print_usage(argv[0]); return 0; }
        else if (!strcmp(a, "--version")) { printf("colibri %s\n", clb_version_string()); return 0; }
        else {
            fprintf(stderr, "unknown option: %s\n", a);
            print_usage(argv[0]);
            return 1;
        }
        #undef NEXT
    }

    clb_init();

    if (verbose) clb_log_set_level(CLB_LOG_DEBUG);

    /* selftest 不需要模型文件 */
    if (selftest) {
        int rc = run_selftest();
        clb_shutdown();
        return rc;
    }

    if (!ep.model_path) {
        fprintf(stderr, "error: --model is required (or use --selftest)\n");
        print_usage(argv[0]);
        clb_shutdown();
        return 1;
    }

    if (gp.sampler.temp <= 0.0f) gp.sampler.mode = CLB_SAMPLER_GREEDY;

    clb_engine *eng = NULL;
    clb_status st = clb_engine_create(&ep, &eng);
    if (st != CLB_OK) {
        fprintf(stderr, "error: %s\n", clb_status_str(st));
        clb_shutdown();
        return 2;
    }

    int32_t n_gen = 0;
    st = clb_engine_generate(eng, &gp, stream_cb, NULL, &n_gen);
    fputc('\n', stdout);

    clb_engine_free(eng);
    clb_shutdown();

    if (st != CLB_OK) {
        fprintf(stderr, "error: %s\n", clb_status_str(st));
        return 3;
    }
    fprintf(stderr, "[generated %d tokens]\n", n_gen);
    return 0;
}