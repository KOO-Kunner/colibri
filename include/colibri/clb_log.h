/* SPDX-License-Identifier: MIT */
#ifndef CLB_LOG_H
#define CLB_LOG_H

#include "clb_export.h"

CLB_EXTERN_C_BEGIN

typedef enum clb_log_level {
    CLB_LOG_TRACE = 0,
    CLB_LOG_DEBUG = 1,
    CLB_LOG_INFO  = 2,
    CLB_LOG_WARN  = 3,
    CLB_LOG_ERROR = 4,
    CLB_LOG_NONE  = 5
} clb_log_level;

typedef void (*clb_log_sink_fn)(clb_log_level lvl, const char *msg, void *user);

CLB_API void          clb_log_set_level(clb_log_level lvl);
CLB_API clb_log_level clb_log_get_level(void);
CLB_API void          clb_log_set_sink(clb_log_sink_fn fn, void *user);
CLB_API void          clb_log(clb_log_level lvl, const char *file, int line,
                              const char *fmt, ...) CLB_PRINTF_FMT(4, 5);

#define CLB_LOGT(...) clb_log(CLB_LOG_TRACE, __FILE__, __LINE__, __VA_ARGS__)
#define CLB_LOGD(...) clb_log(CLB_LOG_DEBUG, __FILE__, __LINE__, __VA_ARGS__)
#define CLB_LOGI(...) clb_log(CLB_LOG_INFO,  __FILE__, __LINE__, __VA_ARGS__)
#define CLB_LOGW(...) clb_log(CLB_LOG_WARN,  __FILE__, __LINE__, __VA_ARGS__)
#define CLB_LOGE(...) clb_log(CLB_LOG_ERROR, __FILE__, __LINE__, __VA_ARGS__)

#define CLB_CHECK(cond, ...)                                                  \
    do {                                                                      \
        if (CLB_UNLIKELY(!(cond))) { CLB_LOGE(__VA_ARGS__); }                 \
    } while (0)

CLB_EXTERN_C_END
#endif /* CLB_LOG_H */