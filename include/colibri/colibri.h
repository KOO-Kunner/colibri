/* SPDX-License-Identifier: MIT */
#ifndef COLIBRI_H
#define COLIBRI_H

#include "clb_export.h"
#include "clb_types.h"
#include "clb_error.h"
#include "clb_log.h"
#include "clb_alloc.h"
#include "clb_tensor.h"
#include "clb_threadpool.h"
#include "clb_quant.h"
#include "clb_cpu.h"
#include "clb_ops.h"
#include "clb_tokenizer.h"
#include "clb_model.h"
#include "clb_context.h"
#include "clb_sampler.h"
#include "clb_engine.h"

CLB_EXTERN_C_BEGIN
CLB_API clb_status clb_init(void);
CLB_API void       clb_shutdown(void);
CLB_API const char *clb_version_string(void);
CLB_EXTERN_C_END

#endif /* COLIBRI_H */