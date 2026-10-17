/* SPDX-License-Identifier: MIT */
#include "colibri/clb_log.h"
#include <stdio.h>
#include <stdarg.h>
#include <string.h>

static clb_log_level  g_level = CLB_LOG_WARN;
static clb_log_sink_fn g_sink = NULL;
static void           *g_sink_user = NULL;

static const char *level_tag(clb_log_level l)
{
    switch (l) {
    case CLB_LOG_TRACE: return "T";
    case CLB_LOG_DEBUG: return "D";
    case CLB_LOG_INFO:  return "I";
    case CLB_LOG_WARN:  return "W";
    case CLB_LOG_ERROR: return "E";
    default:            return "?";
    }
}

void clb_log_set_level(clb_log_level lvl) { g_level = lvl; }
clb_log_level clb_log_get_level(void)      { return g_level; }

void clb_log_set_sink(clb_log_sink_fn fn, void *user)
{
    g_sink = fn;
    g_sink_user = user;
}

static const char *basename_of(const char *p)
{
    const char *s = strrchr(p, '/');
#if defined(_WIN32)
    const char *b = strrchr(p, '\\');
    if (b && (!s || b > s)) s = b;
#endif
    return s ? s + 1 : p;
}

void clb_log(clb_log_level lvl, const char *file, int line,
             const char *fmt, ...)
{
    if (lvl < g_level || lvl >= CLB_LOG_NONE) return;

    char msg[2048];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(msg, sizeof(msg), fmt, ap);
    va_end(ap);

    char out[2304];
    snprintf(out, sizeof(out), "[colibri %s] %s:%d: %s",
             level_tag(lvl), basename_of(file), line, msg);

    if (g_sink) {
        g_sink(lvl, out, g_sink_user);
    } else {
        FILE *f = (lvl >= CLB_LOG_WARN) ? stderr : stdout;
        fprintf(f, "%s\n", out);
        fflush(f);
    }
}