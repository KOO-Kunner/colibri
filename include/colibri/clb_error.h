/* SPDX-License-Identifier: MIT */
#ifndef CLB_ERROR_H
#define CLB_ERROR_H

#include "clb_export.h"

CLB_EXTERN_C_BEGIN

typedef enum clb_status {
    CLB_OK                  =  0,
    CLB_ERR_INVALID_ARG     = -1,
    CLB_ERR_OUT_OF_MEMORY   = -2,
    CLB_ERR_IO              = -3,
    CLB_ERR_FORMAT          = -4,
    CLB_ERR_UNSUPPORTED     = -5,
    CLB_ERR_NOT_FOUND       = -6,
    CLB_ERR_CONTEXT_FULL    = -7,
    CLB_ERR_TOKENIZE        = -8,
    CLB_ERR_THREAD          = -9,
    CLB_ERR_STATE           = -10,
    CLB_ERR_INTERNAL        = -99
} clb_status;

CLB_API const char *clb_status_str(clb_status s);

CLB_EXTERN_C_END
#endif /* CLB_ERROR_H */